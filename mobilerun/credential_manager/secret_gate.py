"""Approval gate and destination checks for ``type_secret``.

Everything here works on non-secret facts only: the foreground app, the
browser host, and which node holds the focus. The guarantee is a check of the
last observed state; the device offers no atomic "check and type", so a focus
change between the final read and the keystrokes cannot be excluded.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlsplit

from mobilerun.credential_manager.approval import ApprovalProvider
from mobilerun.credential_manager.credential_manager import SecretSpec

logger = logging.getLogger("mobilerun")

APPROVAL_DENIED = "approval_denied"
APPROVAL_EXPIRED = "approval_expired"
APPROVAL_UNAVAILABLE = "approval_unavailable"
DEFAULT_APPROVAL_KIND = "secret"
_REFUSAL_REASONS = {"denied": APPROVAL_DENIED, "expired": APPROVAL_EXPIRED}

# Browser chrome the engine knows how to read the host from: package ->
# (address bar identifiers, allowed classes of that node). The host is only
# ever read from these nodes, never from page content. ``None`` means no class
# constraint (iOS: only meaningful where a structured tree with a ``URL`` field
# exists; on the XCTest text tree the gate fails closed anyway).
_ANDROID_BAR_CLASSES = frozenset({"android.widget.EditText", "android.widget.TextView"})
_BROWSER_ADDRESS_BARS: Dict[str, Tuple[Tuple[str, ...], Optional[frozenset]]] = {
    "com.android.chrome": (("com.android.chrome:id/url_bar",), _ANDROID_BAR_CLASSES),
    "com.sec.android.app.sbrowser": (
        ("com.sec.android.app.sbrowser:id/location_bar_edit_text",),
        _ANDROID_BAR_CLASSES,
    ),
    "org.mozilla.firefox": (
        ("org.mozilla.firefox:id/mozac_browser_toolbar_url_view",),
        _ANDROID_BAR_CLASSES,
    ),
    "com.microsoft.emmx": (("com.microsoft.emmx:id/url_bar",), _ANDROID_BAR_CLASSES),
    "com.brave.browser": (("com.brave.browser:id/url_bar",), _ANDROID_BAR_CLASSES),
    # From the public DuckDuckGo Android sources (omnibar EditText); not yet
    # verified on a device.
    "com.duckduckgo.mobile.android": (
        ("com.duckduckgo.mobile.android:id/omnibarTextInput",),
        _ANDROID_BAR_CLASSES,
    ),
    "com.apple.mobilesafari": (("URL",), None),
}
# Address-bar ids of browsers we do NOT know: seeing one outside web content
# means the target is a browser we cannot bind to a host, so nothing is typed.
_GENERIC_ADDRESS_BAR_IDS = frozenset(
    {
        "url_bar",
        "url_edit_text",
        "location_bar_edit_text",
        "mozac_browser_toolbar_url_view",
        "omnibarTextInput",
        "toolbar_url",
        "url_field",
    }
)
# Containers that hold web page content: nothing inside is browser chrome.
_WEB_CONTENT_CLASS_MARKER = "WebView"
_WEB_CONTENT_ID_SUFFIXES = ("compositor_view_holder", "content_container")

_HOST_RE = re.compile(
    r"^(?=.{1,253}$)[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?"
    r"(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*$"
)
_EDITABLE_CLASS_MARKERS = ("EditText", "TextField", "SearchField")


class SecretGateError(Exception):
    """A guard refused typing. The message never contains secret values."""


class SecretApprovalAborted(SecretGateError):
    """Approval was denied, expired or impossible: the whole run must end."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Target:
    package: str
    host: Optional[str] = None


@dataclass(frozen=True)
class NodeInfo:
    """A raw tree node with where it sits (child-index path, web content)."""

    node: dict
    path: Tuple[int, ...]
    in_web_content: bool


@dataclass
class Observation:
    target: Target
    nodes: List[NodeInfo] = field(default_factory=list)
    focused_editable: List[NodeInfo] = field(default_factory=list)
    address_bars: List[NodeInfo] = field(default_factory=list)


@dataclass(frozen=True)
class FieldIdentity:
    """The element the model selected, as seen in the raw pre-tap tree."""

    resource_id: str
    class_name: str
    path: Tuple[int, ...]
    bounds: Tuple[int, int, int, int]


@dataclass
class SecretRunState:
    """Per-run gate state: spec cache and approvals. Rebuilt for every run."""

    specs: Optional[Dict[str, SecretSpec]] = None
    approvals: set = field(default_factory=set)


def run_state(ctx: Any) -> SecretRunState:
    state = getattr(ctx, "secret_run", None)
    if state is None:  # contexts built without ActionContext
        state = ctx.secret_run = SecretRunState()
    return state


async def spec_for(
    state: SecretRunState, manager: Any, secret_id: str
) -> Optional[SecretSpec]:
    """Spec of ``secret_id`` from the per-run cache; ``None`` if unknown."""
    if state.specs is None:
        getter = getattr(manager, "get_secret_specs", None)
        if getter is not None:
            specs = await getter()
        elif hasattr(manager, "get_keys"):  # managers that predate SecretSpec
            specs = [SecretSpec(key=k) for k in await manager.get_keys()]
        else:
            specs = []
        # A failing get_secret_specs() propagates: no spec means no gate, so
        # an unreadable policy must not turn into "typed without approval".
        state.specs = {spec.key: spec for spec in specs}
    return state.specs.get(secret_id)


# ---------------------------------------------------------------------------
# Observation of the destination
# ---------------------------------------------------------------------------


def _is_web_content(node: dict) -> bool:
    class_name = node.get("className") or node.get("class") or ""
    if isinstance(class_name, str) and _WEB_CONTENT_CLASS_MARKER in class_name:
        return True
    return any(i.endswith(_WEB_CONTENT_ID_SUFFIXES) for i in _node_ids(node))


def _walk(
    node: Any, path: Tuple[int, ...] = (), in_web: bool = False
) -> Iterator[NodeInfo]:
    """Depth-first nodes with their path; ``in_web`` tracks ancestors."""
    if isinstance(node, list):
        for i, child in enumerate(node):
            yield from _walk(child, path + (i,), in_web)
    elif isinstance(node, dict):
        yield NodeInfo(node, path, in_web)
        below = in_web or _is_web_content(node)
        for i, child in enumerate(node.get("children") or []):
            yield from _walk(child, path + (i,), below)


def _flag(node: dict, *keys: str) -> bool:
    return any(node.get(key) is True or node.get(key) == "true" for key in keys)


def _is_editable(node: dict) -> bool:
    """Positive evidence that a node is a text field (never keyboard state)."""
    class_name = node.get("className") or node.get("class") or ""
    return _flag(node, "isEditable", "editable") or any(
        marker in class_name for marker in _EDITABLE_CLASS_MARKERS
    )


def _node_ids(node: dict) -> List[str]:
    return [
        value
        for key in ("resourceId", "identifier", "name")
        if isinstance((value := node.get(key)), str) and value
    ]


def _node_text(node: dict) -> str:
    for key in ("text", "value", "label", "contentDescription"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def node_bounds(node: dict) -> Optional[Tuple[int, int, int, int]]:
    raw = node.get("boundsInScreen")
    if isinstance(raw, dict):
        try:
            return (
                int(raw["left"]),
                int(raw["top"]),
                int(raw["right"]),
                int(raw["bottom"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    raw = node.get("bounds")
    if isinstance(raw, str):
        try:
            left, top, right, bottom = (int(part) for part in raw.split(","))
            return left, top, right, bottom
        except ValueError:
            return None
    return None


def host_from_text(text: str) -> Optional[str]:
    """Normalized host of an address-bar text, or ``None`` if it is not one."""
    candidate = text.strip()
    if not candidate or re.search(r"\s", candidate):
        return None
    if "://" in candidate:
        candidate = candidate.split("://", 1)[1]
    try:
        host = urlsplit("//" + candidate).hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.rstrip(".").lower()
    if not _HOST_RE.match(host):
        return None
    if "." not in host and host != "localhost":
        return None
    return host


def observe_target(data: Any) -> Observation:
    """Parse a raw ``driver.get_ui_tree()`` payload into the target facts.

    Raises ``SecretGateError`` when the destination cannot be established.
    """
    if not isinstance(data, dict):
        raise SecretGateError("Cannot read the current screen state.")
    phone_state = data.get("phone_state")
    phone_state = phone_state if isinstance(phone_state, dict) else {}
    tree = data.get("a11y_tree")
    nodes = list(_walk(tree)) if isinstance(tree, (list, dict)) else []

    package = ""
    for key in ("packageName", "package", "bundle_id"):
        value = phone_state.get(key)
        if isinstance(value, str) and value:
            package = value
            break
    if not package:
        raise SecretGateError("Cannot determine the foreground app.")

    adapter = _BROWSER_ADDRESS_BARS.get(package)
    host: Optional[str] = None
    address_bars: List[NodeInfo] = []
    if adapter is not None:
        ids, classes = adapter
        # Browser chrome only: not inside web content, and the right class.
        address_bars = [
            n
            for n in nodes
            if not n.in_web_content
            and any(i in ids for i in _node_ids(n.node))
            and (
                classes is None
                or (n.node.get("className") or n.node.get("class")) in classes
            )
        ]
        hosts = {host_from_text(_node_text(n.node)) for n in address_bars}
        if not address_bars or len(hosts) != 1 or None in hosts:
            raise SecretGateError(
                "Cannot verify the website address in the browser; nothing was typed."
            )
        host = next(iter(hosts))
    else:
        for info in nodes:
            if info.in_web_content:
                continue
            for node_id in _node_ids(info.node):
                if node_id.split(":id/")[-1] in _GENERIC_ADDRESS_BAR_IDS:
                    raise SecretGateError(
                        "This browser is not supported for approved secrets; "
                        "nothing was typed."
                    )

    focused = [
        n
        for n in nodes
        if _flag(n.node, "focused", "isFocused") and _is_editable(n.node)
    ]
    return Observation(Target(package, host), nodes, focused, address_bars)


async def raw_observation(ctx: Any) -> Observation:
    """Fresh destination facts straight from the driver (never shown to the LLM)."""
    driver = ctx.driver
    reader = getattr(driver, "get_ui_tree_unredacted", None) or getattr(
        driver, "get_ui_tree", None
    )
    if reader is None:
        raise SecretGateError("The device cannot report its screen state.")
    return observe_target(await reader())


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


def _approval_key(spec: SecretSpec, target: Target) -> tuple:
    return (
        spec.approval_kind or DEFAULT_APPROVAL_KIND,
        spec.approval_label or spec.key,
        target.package,
        target.host,
    )


async def ensure_approval(ctx: Any, spec: SecretSpec, target: Target) -> None:
    """Make sure this run holds an approval for ``spec`` at ``target``.

    One approval covers every field of the same kind/label/target and nothing
    else. Raises ``SecretApprovalAborted`` unless approved.
    """
    state = run_state(ctx)
    key = _approval_key(spec, target)
    if key in state.approvals:
        return
    provider: Optional[ApprovalProvider] = getattr(ctx, "approval_provider", None)
    if provider is None:
        raise SecretApprovalAborted(APPROVAL_UNAVAILABLE)

    kind, label, package, host = key
    details = {"label": label, "secretGroup": label, "targetPackage": package}
    if host:
        details["targetHost"] = host
    try:
        decision: Any = await provider.request(kind, details)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # the provider is host code; fail closed
        logger.warning("Approval request failed: %s", type(exc).__name__)
        decision = None

    if decision == "approved":
        state.approvals.add(key)
        return
    raise SecretApprovalAborted(
        _REFUSAL_REASONS.get(decision, APPROVAL_UNAVAILABLE)
        if isinstance(decision, str)
        else APPROVAL_UNAVAILABLE
    )


# ---------------------------------------------------------------------------
# Destination checks
# ---------------------------------------------------------------------------


def _element_signature(element: Optional[dict]) -> Optional[tuple]:
    if element is None:
        return None
    return (
        element.get("resourceId"),
        element.get("className"),
        element.get("bounds"),
    )


def ensure_same_element(observed_ui: Any, fresh_ui: Any, index: int) -> dict:
    """The element the model chose must still be the same one in fresh state."""
    fresh = fresh_ui.get_element(index)
    if fresh is None:
        raise SecretGateError(f"Element {index} is no longer on screen.")
    if observed_ui is not None:
        before = observed_ui.get_element(index)
        if before is not None and _element_signature(before) != _element_signature(
            fresh
        ):
            raise SecretGateError(
                "The screen changed since it was last observed; nothing was typed."
            )
    return fresh


def _short_class(name: Any) -> str:
    return name.split(".")[-1] if isinstance(name, str) else ""


def identify_selected(observation: Observation, element: dict) -> FieldIdentity:
    """Pin the selected element to exactly one editable node of the raw tree.

    Fails closed when the element matches no node, several nodes, or a node
    without positive text-field evidence.
    """
    wanted_bounds = node_bounds(element)
    matches = [
        n
        for n in observation.nodes
        if (n.node.get("resourceId") or "") == (element.get("resourceId") or "")
        and _short_class(n.node.get("className") or n.node.get("class"))
        == _short_class(element.get("className"))
        and node_bounds(n.node) == wanted_bounds
        and wanted_bounds is not None
    ]
    if len(matches) != 1:
        raise SecretGateError(
            "Cannot identify the selected element uniquely; nothing was typed."
        )
    info = matches[0]
    if not _is_editable(info.node):
        raise SecretGateError("The selected element is not a text field.")
    return FieldIdentity(
        resource_id=info.node.get("resourceId") or "",
        class_name=_short_class(info.node.get("className") or info.node.get("class")),
        path=info.path,
        bounds=wanted_bounds,
    )


def verify_destination(
    observation: Observation,
    approved: Target,
    *,
    identity: Optional[FieldIdentity] = None,
    tap_point: Optional[Tuple[int, int]] = None,
) -> None:
    """Last check before typing. Raises ``SecretGateError`` unless it holds.

    With an ``identity`` (the field was selected and tapped) the focused node
    must be exactly that element: identity first (resource id, class, tree
    path), geometry as an additional check.
    """
    if observation.target != approved:
        raise SecretGateError(
            "The target app or website changed after approval; nothing was typed."
        )
    if len(observation.focused_editable) != 1:
        raise SecretGateError(
            "No single focused text field was found; nothing was typed."
        )
    focused = observation.focused_editable[0]
    if any(focused is bar for bar in observation.address_bars):
        raise SecretGateError("Refusing to type a secret into the address bar.")
    if identity is None:
        return
    node = focused.node
    mismatch = SecretGateError(
        "The focused field is not the one that was selected; nothing was typed."
    )
    if (
        (node.get("resourceId") or "") != identity.resource_id
        or _short_class(node.get("className") or node.get("class"))
        != identity.class_name
        or focused.path != identity.path
    ):
        raise mismatch
    focused_bounds = node_bounds(node)
    if focused_bounds is None:
        raise SecretGateError("Cannot verify the focused field; nothing was typed.")
    if tap_point is not None and not _contains(focused_bounds, tap_point):
        raise mismatch
    if not (
        _nested(focused_bounds, identity.bounds)
        or _nested(identity.bounds, focused_bounds)
    ):
        raise mismatch


def _contains(bounds: Tuple[int, int, int, int], point: Tuple[int, int]) -> bool:
    left, top, right, bottom = bounds
    x, y = point
    return left <= x < right and top <= y < bottom


def _nested(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )
