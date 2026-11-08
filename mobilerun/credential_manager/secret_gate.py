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

from mobilerun.credential_manager.approval import ApprovalDecision, ApprovalProvider
from mobilerun.credential_manager.credential_manager import SecretSpec

logger = logging.getLogger("mobilerun")

APPROVAL_DENIED = "approval_denied"
APPROVAL_EXPIRED = "approval_expired"
APPROVAL_UNAVAILABLE = "approval_unavailable"
DEFAULT_APPROVAL_KIND = "secret"

# Browser chrome the engine knows how to read the host from: package ->
# identifiers of its address bar node. The host is only ever read from these
# nodes, never from page content.
_BROWSER_ADDRESS_BARS: Dict[str, Tuple[str, ...]] = {
    "com.android.chrome": ("com.android.chrome:id/url_bar",),
    "com.sec.android.app.sbrowser": (
        "com.sec.android.app.sbrowser:id/location_bar_edit_text",
    ),
    "org.mozilla.firefox": ("org.mozilla.firefox:id/mozac_browser_toolbar_url_view",),
    "com.microsoft.emmx": ("com.microsoft.emmx:id/url_bar",),
    "com.brave.browser": ("com.brave.browser:id/url_bar",),
    "com.apple.mobilesafari": ("URL",),
}
# Address-bar ids of browsers we do NOT know: seeing one means the target is a
# web context we cannot bind to a host, so nothing is typed.
_GENERIC_ADDRESS_BAR_IDS = frozenset(
    {
        "url_bar",
        "url_edit_text",
        "location_bar_edit_text",
        "mozac_browser_toolbar_url_view",
        "toolbar_url",
        "url_field",
    }
)

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


@dataclass
class Observation:
    target: Target
    focused_editable: List[dict] = field(default_factory=list)
    address_bars: List[dict] = field(default_factory=list)


@dataclass
class SecretRunState:
    """Per-run gate state: spec cache and approvals. Rebuilt for every run."""

    specs: Optional[Dict[str, SecretSpec]] = None
    approvals: set = field(default_factory=set)


def run_state(ctx: Any) -> SecretRunState:
    state = getattr(ctx, "secret_run", None)
    if state is None:
        state = SecretRunState()
        try:
            ctx.secret_run = state
        except AttributeError:
            pass
    return state


async def spec_for(ctx: Any, secret_id: str) -> Optional[SecretSpec]:
    """Spec of ``secret_id`` from the per-run cache; ``None`` if unknown."""
    state = run_state(ctx)
    if state.specs is None:
        manager = ctx.credential_manager
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


def _walk(node: Any) -> Iterator[dict]:
    if isinstance(node, list):
        for child in node:
            yield from _walk(child)
    elif isinstance(node, dict):
        yield node
        yield from _walk(node.get("children") or [])


def _flag(node: dict, *keys: str) -> bool:
    return any(node.get(key) is True or node.get(key) == "true" for key in keys)


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

    known_ids = _BROWSER_ADDRESS_BARS.get(package)
    host: Optional[str] = None
    address_bars: List[dict] = []
    if known_ids is not None:
        address_bars = [n for n in nodes if any(i in known_ids for i in _node_ids(n))]
        hosts = {host_from_text(_node_text(n)) for n in address_bars}
        if not address_bars or len(hosts) != 1 or None in hosts:
            raise SecretGateError(
                "Cannot verify the website address in the browser; nothing was typed."
            )
        host = next(iter(hosts))
    else:
        for node in nodes:
            for node_id in _node_ids(node):
                if node_id.split(":id/")[-1] in _GENERIC_ADDRESS_BAR_IDS:
                    raise SecretGateError(
                        "This browser is not supported for approved secrets; "
                        "nothing was typed."
                    )

    keyboard_open = phone_state.get("isEditable") is True
    focused = [
        n
        for n in nodes
        if _flag(n, "focused", "isFocused")
        and (
            _flag(n, "isEditable", "editable")
            or any(
                m in (n.get("className") or n.get("class") or "")
                for m in _EDITABLE_CLASS_MARKERS
            )
            or keyboard_open
        )
    ]
    return Observation(Target(package, host), focused, address_bars)


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
    decision: Any
    try:
        decision = await provider.request(kind, details)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # the provider is host code; fail closed
        logger.warning("Approval request failed: %s", type(exc).__name__)
        decision = None

    decision_: Optional[ApprovalDecision] = decision
    if decision_ == "approved":
        state.approvals.add(key)
        return
    reason = {"denied": APPROVAL_DENIED, "expired": APPROVAL_EXPIRED}.get(
        decision_ if isinstance(decision_, str) else "", APPROVAL_UNAVAILABLE
    )
    raise SecretApprovalAborted(reason)


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


def verify_destination(
    observation: Observation,
    approved: Target,
    *,
    selected_bounds: Optional[Tuple[int, int, int, int]] = None,
    tap_point: Optional[Tuple[int, int]] = None,
) -> None:
    """Last check before typing. Raises ``SecretGateError`` unless it holds."""
    if observation.target != approved:
        raise SecretGateError(
            "The target app or website changed after approval; nothing was typed."
        )
    if len(observation.focused_editable) != 1:
        raise SecretGateError(
            "No single focused text field was found; nothing was typed."
        )
    field_node = observation.focused_editable[0]
    if any(field_node is bar for bar in observation.address_bars):
        raise SecretGateError("Refusing to type a secret into the address bar.")
    if selected_bounds is None:
        return
    focused_bounds = node_bounds(field_node)
    if focused_bounds is None:
        raise SecretGateError("Cannot verify the focused field; nothing was typed.")
    if tap_point is not None and not _contains(focused_bounds, tap_point):
        raise SecretGateError(
            "The focused field is not the one that was selected; nothing was typed."
        )
    if not (
        _nested(focused_bounds, selected_bounds)
        or _nested(selected_bounds, focused_bounds)
    ):
        raise SecretGateError(
            "The focused field is not the one that was selected; nothing was typed."
        )


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
