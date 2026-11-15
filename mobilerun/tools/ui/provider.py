"""StateProvider — orchestrates fetching and parsing device state.

Fetches raw data from a ``DeviceDriver``, applies tree filters/formatters,
and produces a ``UIState`` snapshot.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from copy import copy, deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

import httpx
from mobilerun_core_local.driver.base import DeviceDisconnectedError

from mobilerun.tools.helpers.coordinate import NORMALIZED_STATE_NOTE
from mobilerun.tools.helpers.images import (
    fit_dimensions_to_max_side,
    resize_image_to_dimensions_with_grid,
    resize_image_to_max_side_with_grid,
)
from mobilerun.tools.ui.state import UIState
from mobilerun.tools.ui.stealth_state import StealthUIState

if TYPE_CHECKING:
    from mobilerun_core_local.driver.base import DeviceDriver

    from mobilerun.agent.action_context import ActionContext
    from mobilerun.agent.selectors.base import ActionCandidate
    from mobilerun.tools.filters import TreeFilter
    from mobilerun.tools.formatters import TreeFormatter

logger = logging.getLogger("mobilerun")

# Retry schedule: delay in seconds after each failed attempt.
# Total wait across 7 attempts: 1+2+3+5+8+10 = 29s.
_RETRY_DELAYS = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
_MAX_RETRIES = 7

# After this many consecutive failures, run the recovery callback.
# With the schedule above, this fires after ~11s (1+2+3+5).
_RECOVERY_AFTER_ATTEMPT = 5

# A timed-out read means the device side is still working on the request we
# just abandoned. Portal and WDA serve state one request at a time, so every
# extra retry queues behind the stuck one. Give timeouts a single retry
# (preceded by recovery), not the full schedule above.
_MAX_TIMEOUT_ATTEMPTS = 2

# Say we are waiting when one fetch takes longer than this. A slow app can
# keep a state request busy for 15-30s, and that should not look like a hang.
_SLOW_FETCH_NOTICE_S = 3.0

# Portal's error when its accessibility service is not bound, e.g. after the
# system force-stopped Portal. Waiting does not help, so recover right away.
_A11Y_SERVICE_NOT_AVAILABLE = "Accessibility service not available"


def _is_timeout(error: BaseException) -> bool:
    """True when the device accepted the request but did not answer in time.

    Connect and pool timeouts are excluded: those requests never reached the
    device, so retrying them does not queue work behind a busy one.
    """
    return isinstance(error, (TimeoutError, httpx.ReadTimeout))


async def _fetch_with_notice(
    fetch: Callable[[], Awaitable[Dict[str, Any]]], notice: bool
) -> Dict[str, Any]:
    """Await *fetch*, logging once at INFO if it runs past the notice delay."""
    if not notice:
        return await fetch()
    task = asyncio.ensure_future(fetch())
    try:
        done, _ = await asyncio.wait({task}, timeout=_SLOW_FETCH_NOTICE_S)
    except BaseException:
        task.cancel()
        raise
    if not done:
        logger.info("Waiting for the UI state from the device...")
    return await task


async def fetch_state_with_retry(
    fetch: Callable[[], Awaitable[Dict[str, Any]]],
    recovery: Optional[Callable[[], Awaitable[None]]] = None,
    max_retries: int = _MAX_RETRIES,
    retry_delays: Optional[List[float]] = None,
    recovery_after: int = _RECOVERY_AFTER_ATTEMPT,
) -> Dict[str, Any]:
    """Fetch raw device state with retries, backoff, and mid-retry recovery.

    Args:
        fetch: Async callable that returns the raw state dict from Portal.
        recovery: Optional async callable invoked once after *recovery_after*
            consecutive failures (e.g. restart accessibility service), or
            after the first failure if Portal's service is not available,
            or before the single retry of a timed-out fetch.
        max_retries: Total number of attempts before giving up. Timeouts
            stop after ``_MAX_TIMEOUT_ATTEMPTS`` timed-out attempts.
        retry_delays: Per-attempt delays. If shorter than max_retries - 1,
            the last value is reused for remaining delays.
        recovery_after: Trigger *recovery* after this many failures.

    Returns:
        The raw state dict (guaranteed to contain ``a11y_tree``,
        ``phone_state``, ``device_context``).

    Raises:
        DeviceDisconnectedError: Re-raised immediately.
        Exception: After all retries are exhausted.
    """
    delays = retry_delays or _RETRY_DELAYS
    last_error: Optional[Exception] = None
    recovery_attempted = False
    slow_notice_logged = False
    timeouts = 0
    attempts = 0

    is_debug = logger.isEnabledFor(logging.DEBUG)

    for attempt in range(max_retries):
        try:
            attempts = attempt + 1
            logger.debug(f"Getting state (attempt {attempts}/{max_retries})")

            t0 = time.monotonic()
            try:
                combined_data = await _fetch_with_notice(fetch, not slow_notice_logged)
            finally:
                if time.monotonic() - t0 >= _SLOW_FETCH_NOTICE_S:
                    slow_notice_logged = True

            if is_debug:
                elapsed = (time.monotonic() - t0) * 1000
                logger.debug(f"State fetched in {elapsed:.0f}ms")

            if "error" in combined_data:
                raise Exception(f"Portal returned error: {combined_data}")

            required_keys = ["a11y_tree", "phone_state", "device_context"]
            missing = [k for k in required_keys if k not in combined_data]
            if missing:
                raise Exception(f"Missing data in state: {', '.join(missing)}")

            return combined_data

        except DeviceDisconnectedError:
            raise
        except Exception as e:
            last_error = e
            timed_out = _is_timeout(e)
            if timed_out:
                timeouts += 1
            is_last = attempt >= max_retries - 1 or timeouts >= _MAX_TIMEOUT_ATTEMPTS
            delay = delays[attempt] if attempt < len(delays) else delays[-1]

            err_desc = str(e) or type(e).__name__
            suffix = f" (retrying in {delay:.0f}s)" if not is_last else ""
            logger.warning(
                f"get_state attempt {attempt + 1} failed: {err_desc}{suffix}"
            )

            # Mid-retry recovery: restart the a11y service once
            service_gone = _A11Y_SERVICE_NOT_AVAILABLE in err_desc
            if (
                not recovery_attempted
                and recovery is not None
                and (attempt + 1 >= recovery_after or service_gone or timed_out)
                and not is_last
            ):
                recovery_attempted = True
                if service_gone:
                    logger.info(
                        "Portal accessibility service is off, re-enabling it..."
                    )
                else:
                    logger.info("State retrieval failing, attempting recovery...")
                try:
                    await recovery()
                    logger.info("Recovery action completed")
                except Exception as rec_err:
                    logger.warning(f"Recovery action failed: {rec_err}")

            if is_last:
                break
            await asyncio.sleep(delay)

    last_desc = str(last_error) or type(last_error).__name__
    error_msg = f"Failed to get state after {attempts} attempts: {last_desc}"
    logger.error(error_msg)
    raise Exception(error_msg) from last_error


class StateProvider:
    """Base class — subclass to support different platforms."""

    supported: set[str] = set()
    # True when coordinates a model derives from the screenshot map to the
    # driver's tap input space — either directly (Android: screenshot and
    # input coords are both device pixels) or via ``convert_point`` scaling
    # (the vision coordinate contract; iOS sets this only when the contract
    # is active, since its screenshots are physical pixels while taps use
    # XCTest points). Gates the click_at auto-unmask with vision.
    screenshot_matches_input_coords: bool = False
    # True when the screenshot attached to LLM messages must first be resized
    # (with a labeled coordinate grid) into the deterministic coordinate space
    # the provider declared in ``formatted_text`` and compensates for via
    # ``coordinate_scale_x/y``. Vision-model output spaces are otherwise
    # model-dependent: a raw native screenshot is downscaled inside the LLM
    # provider's API and different models answer in different spaces.
    resize_model_screenshot: bool = False

    def __init__(self, driver: "DeviceDriver") -> None:
        self.driver = driver

    async def get_state(self) -> UIState:
        raise NotImplementedError

    async def get_final_state(self) -> UIState:
        """Read the final observation; subclasses may make this narrower."""
        return await self.get_state()

    def get_action_candidates(self, state: UIState) -> tuple[ActionCandidate, ...]:
        """Enumerate deterministic element actions; providers may override this."""
        from mobilerun.agent.fast_agent.xml_parser import ToolCall
        from mobilerun.agent.selectors.base import ActionCandidate

        options = []
        seen = set()
        elements = UIState._collect_all(state.elements)
        screen_area = (state.screen_width or 0) * (state.screen_height or 0)
        borrowed = _borrowed_details(elements, screen_area)
        field_labels = {
            id(element): label
            for element in elements
            if _kind(element) in _TEXT_KINDS
            and not (element.get("details") or {}).get("description")
            and not (element.get("details") or {}).get("hint")
            and (bounds := _parse_bounds(element)) is not None
            and (label := _label_beside(element, bounds, elements))
        }
        toggle_names, duplicates = _toggle_names(elements, borrowed, screen_area)
        duplicates |= _same_named_wrappers(
            elements, borrowed, toggle_names, field_labels
        )
        hidden, submit_keys = _hidden_ios_elements(elements, screen_area)
        duplicates |= hidden
        focused = _focused_element(state.phone_state, elements)
        # One tap target per bounds: iOS reports a cell and the button filling
        # it as two elements. Keep the best-named one, in tree order.
        targets: dict[str, list] = {}  # bounds -> [rank, order, {tool: option}]
        for element in elements:
            index = element.get("index")
            if (
                type(index) is not int
                or index in seen
                or element.get("isEnabled") is False
                or element.get("isVisibleToUser") is False
            ):
                continue
            seen.add(index)
            if id(element) in duplicates:
                continue
            bounds = element.get("bounds")
            if not isinstance(bounds, str):
                continue
            try:
                left, top, right, bottom = map(int, bounds.split(","))
                if right <= left or bottom <= top:
                    continue
                center = ((left + right) // 2, (top + bottom) // 2)
                # Clicks tap the centre, so it must be on screen.
                if state.screen_width and not 0 <= center[0] < state.screen_width:
                    continue
                if state.screen_height and not 0 <= center[1] < state.screen_height:
                    continue
                UIState._avoid_tap_blockers(state, element, center)
            except (KeyError, TypeError, ValueError):
                continue
            label = _candidate_label(
                element,
                borrowed,
                toggle_names,
                focused,
                center[1],
                state.screen_height,
                field_labels,
                submit_keys,
            )
            element_options = []
            # Text fields focus on tap even when not marked clickable (iOS).
            if element.get("isClickable", True) or _kind(element) in _TEXT_KINDS:
                element_options.append(
                    ActionCandidate(
                        ToolCall("click", {"index": index}), f"Click {label}"
                    )
                )
            if element.get("isLongClickable", False):
                element_options.append(
                    ActionCandidate(
                        ToolCall("long_press", {"index": index}), f"Long press {label}"
                    )
                )
            if not element_options:
                continue
            rank = _name_rank(element, borrowed, toggle_names)
            if rank == 0 and _is_ios(element) and _wraps_option(element, elements):
                continue  # An unnamed iOS cell around its own button or switch.
            by_tool = {option.action.name: option for option in element_options}
            kept = targets.get(bounds)
            if kept is None:
                targets[bounds] = [rank, len(targets), by_tool]
            elif rank > kept[0]:
                # A twin may long-press where the better-named element only clicks.
                targets[bounds] = [rank, kept[1], {**kept[2], **by_tool}]
            else:
                kept[2] = {**by_tool, **kept[2]}
        for _, _, by_tool in sorted(targets.values(), key=lambda t: t[1]):
            options.extend(
                sorted(by_tool.values(), key=lambda o: o.action.name != "click")
            )
        return tuple(options)

    async def build_action_candidates(
        self,
        ctx: ActionContext,
        tools: dict[str, Any],
        *,
        include_device_actions: bool = False,
        locate_taps: bool = False,
    ) -> tuple[ActionCandidate, ...]:
        """Keep provider overrides and add only enabled, supported device actions.

        ``locate_taps`` offers a tap whose point the selector locates on the
        screenshot, for screens without elements to choose from.
        """
        from mobilerun.agent.fast_agent.xml_parser import ToolCall
        from mobilerun.agent.selectors.base import ActionCandidate

        options = list(self.get_action_candidates(ctx.ui)) if ctx.ui is not None else []
        if locate_taps and "click_at" in tools:
            options.extend(self._tap_candidates(ctx.ui))
        if not include_device_actions:
            return tuple(options)

        if "system_button" in tools:
            navigation = {
                "back",
                "home",
                "enter",
                "recent",
                "recents",
                "app_switch",
                "overview",
            }
            for button in sorted(
                getattr(self.driver, "supported_buttons", set()) & navigation
            ):
                options.append(
                    ActionCandidate(
                        ToolCall("system_button", {"button": button}),
                        f"Press {button}: "
                        + {
                            "back": "go back or dismiss the keyboard/dialog",
                            "home": "go to the home screen without closing apps",
                            "enter": "submit the focused field",
                        }.get(button, "show recent apps"),
                    )
                )

        if "swipe" in tools:
            options.extend(self._swipe_candidates(ctx.ui))
        if "wait" in tools:
            options.append(
                ActionCandidate(
                    ToolCall("wait", {"duration": 1.0}),
                    "Wait 1 second for a loading screen, animation, or UI update",
                )
            )
        if "open_app" in tools and "get_apps" in getattr(
            self.driver, "supported", set()
        ):
            options.extend(await self._app_candidates(ctx, tools["open_app"]))

        # A provider may already supply some device actions. Preserve its order and
        # avoid duplicate choices without touching the executable parameters.
        unique: list[ActionCandidate] = []
        for option in options:
            if option.action.name in tools and not any(
                option.action == item.action for item in unique
            ):
                unique.append(option)
        return tuple(unique)

    def _action_space(self, state: UIState | None) -> tuple[float, float] | None:
        """Size of the coordinate space coordinate actions are written in."""
        if state is None or not state.screen_width or not state.screen_height:
            return None
        if (
            getattr(self, "requires_active_contract_for_coords", False)
            and not state.coordinate_contract_active
        ):
            return None
        if state.use_normalized:
            width = height = 1000
        elif getattr(self, "requires_coordinate_tools", False):
            width, height = state.screen_width, state.screen_height
        else:
            try:
                width = state.screen_width / state.coordinate_scale_x
                height = state.screen_height / state.coordinate_scale_y
            except (TypeError, ValueError, ZeroDivisionError):
                return None
        if not all(math.isfinite(value) and value >= 4 for value in (width, height)):
            return None
        return width, height

    def _tap_candidates(self, state: UIState | None) -> list[ActionCandidate]:
        """Screenshot-only screens have no elements; the model locates the tap."""
        from mobilerun.agent.fast_agent.xml_parser import ToolCall
        from mobilerun.agent.selectors.base import ActionCandidate

        space = self._action_space(state)
        if not getattr(self, "requires_coordinate_tools", False) or space is None:
            return []
        return [
            ActionCandidate(
                ToolCall("click_at"),
                "Tap an element on the screen (located on the screenshot)",
                point_space=space,
            )
        ]

    def _swipe_candidates(self, state: UIState | None) -> list[ActionCandidate]:
        from mobilerun.agent.fast_agent.xml_parser import ToolCall
        from mobilerun.agent.selectors.base import ActionCandidate

        space = self._action_space(state)
        if space is None:
            return []
        width, height = space

        def point(x: float, y: float) -> list[int]:
            return [round((width - 1) * x), round((height - 1) * y)]

        return [
            ActionCandidate(
                ToolCall(
                    "swipe",
                    {
                        "coordinate": point(*start),
                        "coordinate2": point(*end),
                        "duration": 0.5,
                    },
                ),
                description,
            )
            for start, end, description in (
                (
                    (0.5, 0.8),
                    (0.5, 0.2),
                    "Swipe up to scroll down or reveal apps from the home screen",
                ),
                ((0.5, 0.2), (0.5, 0.8), "Swipe down to scroll up"),
                ((0.8, 0.5), (0.2, 0.5), "Swipe left to the next page or panel"),
                ((0.2, 0.5), (0.8, 0.5), "Swipe right to the previous page or panel"),
            )
        ]

    async def _app_candidates(
        self,
        ctx: ActionContext,
        signature: dict[str, Any],
    ) -> list[ActionCandidate]:
        from mobilerun.agent.action_context import ActionCandidateApp
        from mobilerun.agent.fast_agent.xml_parser import ToolCall
        from mobilerun.agent.selectors.base import ActionCandidate

        params = signature.get("parameters", {})
        parameter = next(
            (name for name in ("app_id", "bundle_id", "text") if name in params), None
        )
        if parameter is None:
            return []
        if ctx.action_candidate_apps is None:
            try:
                apps = await self.driver.get_apps(include_system=True)
            except DeviceDisconnectedError:
                raise
            except Exception as exc:
                logger.warning("Could not list apps for decision candidates: %s", exc)
                return []
            installed: dict[str, ActionCandidateApp] = {}
            for app in apps:
                if not isinstance(app, dict) or app.get("launchable") is False:
                    continue
                identifier = next(
                    (
                        app.get(key)
                        for key in (
                            "package",
                            "package_name",
                            "bundle_id",
                            "app_id",
                        )
                        if isinstance(app.get(key), str) and app[key].strip()
                    ),
                    None,
                )
                if identifier:
                    label = app.get("label") or app.get("name") or identifier
                    installed[identifier] = ActionCandidateApp(
                        identifier=identifier, label=str(label)
                    )
            ctx.action_candidate_apps = tuple(
                sorted(
                    installed.values(),
                    key=lambda app: (app.label.casefold(), app.identifier),
                )
            )

        current_app = (
            ctx.ui.phone_state.get("packageName") if ctx.ui is not None else None
        )
        goal = str(getattr(ctx.shared_state, "instruction", "") or "").casefold()
        # Apps named in the goal go first so a capped option list keeps them.
        apps = sorted(
            ctx.action_candidate_apps,
            key=lambda app: not (len(app.label) > 1 and app.label.casefold() in goal),
        )
        return [
            ActionCandidate(
                ToolCall("open_app", {parameter: app.identifier}),
                f"Open app {app.label} ({app.identifier})",
            )
            for app in apps
            if app.identifier != current_app
        ]


_MAX_BORROWED_TEXTS = 3
_MAX_OWNER_SCREEN_SHARE = 0.5
# Share of the screen height that counts as its top or bottom edge.
_EDGE_SHARE = 0.1

_KINDS = {
    "switch": "switch",
    "switchcompat": "switch",
    "switchmaterial": "switch",
    "materialswitch": "switch",
    "toggle": "switch",
    "togglebutton": "switch",
    "checkbox": "checkbox",
    "materialcheckbox": "checkbox",
    # Used for both multi-choice and single-choice (radio) list rows.
    "checkedtextview": "option",
    "radiobutton": "radio button",
    "materialradiobutton": "radio button",
    "button": "button",
    "imagebutton": "button",
    "materialbutton": "button",
    "appcompatbutton": "button",
    "appcompatimagebutton": "button",
    "floatingactionbutton": "button",
    "extendedfloatingactionbutton": "button",
    "edittext": "text field",
    "appcompatedittext": "text field",
    "textinputedittext": "text field",
    "autocompletetextview": "text field",
    "textfield": "text field",
    "securetextfield": "text field",
    "searchfield": "search field",
    "searchview": "search field",
    "seekbar": "slider",
    "slider": "slider",
    "spinner": "dropdown",
    "tab": "tab",
}

_TEXT_KINDS = {"text field", "search field"}
_TOGGLE_KINDS = {"switch", "checkbox", "radio button", "option"}
# mobilerun-ios sends XCUITest class names; WDA fills empty texts with them.
_IOS_CLASS_PREFIX = "XCUIElementType"
_SECRET_MARK = "[secret:"


def _class_name(element: dict) -> str:
    return str(element.get("className") or element.get("type") or "")


def _short_class(element: dict) -> str:
    name = _class_name(element).rsplit(".", 1)[-1]
    return name.removeprefix(_IOS_CLASS_PREFIX)


def _kind(element: dict) -> str:
    """A plain word for widget types a model can act on, or ""."""
    short = _short_class(element)
    if short == "TextView" and (
        _class_name(element).startswith(_IOS_CLASS_PREFIX)
        or element.get("type") == "TextView"
    ):
        return "text field"  # iOS TextView is editable; Android's is a label.
    return _KINDS.get(short.lower(), "")


def _is_ios(element: dict) -> bool:
    """iOS toggles report their state as a "0"/"1" text; Android ones can be named so."""
    return _class_name(element).startswith(_IOS_CLASS_PREFIX) or "type" in element


def _ios_value_text(element: dict, text: str) -> bool:
    """iOS puts a toggle's, cell's or tab's 0/1 value in the text; the label
    then sits in the description. A key labelled "1" keeps its text."""
    description = str((element.get("details") or {}).get("description") or "")
    return (
        _kind(element) in _TOGGLE_KINDS
        or _short_class(element) == "Cell"
        or (bool(description) and description != text)
    )


def _toggle_value(element: dict) -> str:
    """iOS reports a toggle's state as its value: "1"/"0"."""
    for key in ("value", "text"):
        value = str(element.get(key) or "").strip()
        if value in {"0", "1"}:
            return value
    return ""


def _check_state(element: dict) -> str:
    """ "on"/"off" for switches, "checked"/"unchecked" for other checkables."""
    kind = _kind(element)
    state = str(element.get("checkedState") or "")
    if state.startswith("isChecked="):
        checked = state == "isChecked=True"
    elif kind in _TOGGLE_KINDS and (value := _toggle_value(element)):
        checked = value == "1"
    else:
        return ""
    if kind == "switch":
        return "on" if checked else "off"
    return "checked" if checked else "unchecked"


def _own_text(element: dict) -> str:
    """Element text, or "" when it only repeats an id, a class or a toggle value."""
    text = str(element.get("text") or "").strip()
    class_name = _class_name(element)
    if (
        not text
        or text == element.get("resourceId")
        or text == class_name
        or ((short := class_name.rsplit(".", 1)[-1]) and text.endswith(f".{short}"))
        or text.startswith(_IOS_CLASS_PREFIX)
        or (
            text in {"0", "1"}
            # Only mobilerun-ios puts values in text; the legacy portal keeps them in value.
            and _class_name(element).startswith(_IOS_CLASS_PREFIX)
            and _ios_value_text(element, text)
        )
    ):
        return ""
    return text


def _display_name(element: dict) -> str:
    """The element's own name: its text, with a description the text lacks."""
    facts = element.get("details") or {}
    own = _own_text(element)
    description = str(facts.get("description") or "").strip()
    if description.startswith(_IOS_CLASS_PREFIX):
        description = ""
    if not own:
        return description if _kind(element) not in _TEXT_KINDS else ""
    if (
        description
        and description.lower() not in own.lower()
        and _kind(element) not in _TEXT_KINDS
    ):
        return f"{own}, {description}"  # e.g. day chip "T, Tuesday"
    return own


def _name_and_contents(
    element: dict, kind: str, name: str, label: str = ""
) -> tuple[str, str | None]:
    """A field's name and what it holds, from wherever the platform reports them.

    ``label`` is the text beside a field that has no name of its own.
    """
    facts = element.get("details") or {}
    description = str(facts.get("description") or "").strip() or label
    hint = str(facts.get("hint") or element.get("placeholder") or "").strip()
    if facts.get("password"):
        return description or hint or name, None
    if kind in _TEXT_KINDS and "type" in element:
        # Legacy iOS portal: text is the label (or placeholder), contents are the value.
        text = _own_text(element)
        own_label = str(element.get("label") or "").strip()
        field_name = label or (own_label if own_label != hint else "") or hint or name
        value = str(element.get("value") or "").strip()
        if value and value not in {hint, field_name}:
            return field_name, f'contains "{value}"'
        return field_name, None
    if kind in _TEXT_KINDS:
        text = _own_text(element)
        field_name = description or hint
        # Empty fields show their hint, or (iOS) fall back to their label.
        if facts.get("showingHint") or (text and text in {description, hint}):
            return field_name or name, None
        if field_name and text:
            return field_name, f'contains "{text}"'
        name = field_name or name
    # iOS debugDescription: the value is reported apart from the label, and
    # an empty field reports its placeholder as its value.
    value = str(element.get("value") or "").strip()
    placeholder = str(element.get("placeholder") or "").strip()
    if value and value not in {name, placeholder} and kind not in _TOGGLE_KINDS:
        if kind in _TEXT_KINDS:
            return name, f'contains "{value}"'
        if kind in {"slider", "dropdown"}:
            return name, value
    return name, None


def _parse_bounds(element: dict) -> tuple[int, int, int, int] | None:
    try:
        left, top, right, bottom = map(int, element["bounds"].split(","))
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    return left, top, right, bottom


def _inside(inner: tuple[int, ...], outer: tuple[int, ...], slack: int = 0) -> bool:
    return (
        outer[0] - slack <= inner[0]
        and outer[1] - slack <= inner[1]
        and inner[2] <= outer[2] + slack
        and inner[3] <= outer[3] + slack
    )


# iOS switch knobs can stick out of their row by a couple of points.
_KNOB_SLACK = 4


def _contains(container: dict, element: dict) -> bool:
    """Whether ``element`` is nested in ``container``, when the tree says so."""
    start, size = container.get("index"), container.get("descendants")
    index = element.get("index")
    if not all(type(value) is int for value in (start, size, index)):
        return True  # No nesting information: bounds decide.
    return start < index <= start + size


def _area(bounds: tuple[int, ...]) -> int:
    return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])


def _focused_element(phone_state: Any, elements: list[dict]) -> int | None:
    """id() of the one element that is the focused editable field, if known."""
    if not isinstance(phone_state, dict) or not phone_state.get("isEditable"):
        return None
    focused = phone_state.get("focusedElement")
    if not isinstance(focused, dict) or not (resource := focused.get("resourceId")):
        return None
    matches = [element for element in elements if element.get("resourceId") == resource]
    if len(matches) > 1 and focused.get("text"):
        # Rows often reuse one id; the focused field's text tells them apart.
        matches = [
            element for element in matches if element.get("text") == focused["text"]
        ]
    return id(matches[0]) if len(matches) == 1 else None


def _is_actionable(element: dict) -> bool:
    return bool(
        element.get("isClickable", True) or element.get("isLongClickable", False)
    )


def _visible(element: dict) -> bool:
    return element.get("isVisibleToUser") is not False


def _offerable(element: dict) -> bool:
    """Whether the element can become an option at all."""
    bounds = _parse_bounds(element)
    return (
        _visible(element)
        and element.get("isEnabled") is not False
        and (_is_actionable(element) or _kind(element) in _TEXT_KINDS)
        and bounds is not None
        and bounds[2] > bounds[0]
        and bounds[3] > bounds[1]
    )


def _borrowed_details(elements: list[dict], screen_area: int) -> dict[int, dict]:
    """Texts and widget states inside each clickable container.

    The flattened tree drops hierarchy: list rows are often text-less
    containers whose title, and switch, are separate elements within their
    bounds. Each text belongs to its innermost clickable container; screen
    roots, scrims and overlays are not rows, so they borrow nothing.
    """
    clickables = []
    for element in elements:
        if _is_actionable(element) and (bounds := _parse_bounds(element)) is not None:
            if (
                not screen_area
                or _area(bounds) <= screen_area * _MAX_OWNER_SCREEN_SHARE
            ):
                clickables.append((_area(bounds), bounds, element))
    borrowed: dict[int, dict] = {}
    for element in elements:
        if not _visible(element) or (bounds := _parse_bounds(element)) is None:
            continue
        if _is_actionable(element):
            continue  # It is an option of its own, with its own name and state.
        text = "" if _kind(element) in _TEXT_KINDS else _display_name(element)
        state = _check_state(element)
        if not (text or state):
            continue
        owner = min(
            (
                (area, container)
                for area, container_bounds, container in clickables
                if container is not element
                and _inside(bounds, container_bounds)
                and _contains(container, element)
            ),
            key=lambda item: item[0],
            default=None,
        )
        if owner is None:
            continue
        details = borrowed.setdefault(id(owner[1]), {"texts": [], "widget": None})
        texts = details["texts"]
        if text and text not in texts and len(texts) < _MAX_BORROWED_TEXTS:
            texts.append(text)
        if state and details["widget"] is None:
            details["widget"] = (_kind(element), state)
    return borrowed


def _name_rank(
    element: dict, borrowed: dict[int, dict], toggle_names: dict[int, str]
) -> int:
    """2 for an own name, 1 for a name borrowed from inside, 0 for none."""
    if (
        _display_name(element)
        or toggle_names.get(id(element))
        or (_kind(element) in _TEXT_KINDS and _own_text(element))
    ):
        return 2
    return 1 if (borrowed.get(id(element)) or {}).get("texts") else 0


def _wraps_option(container: dict, elements: list[dict]) -> bool:
    """Whether a nested element is an option of its own."""
    if type(container.get("descendants")) is not int:
        return False
    return any(
        element is not container
        and _contains(container, element)
        and _offerable(element)
        for element in elements
    )


def _toggle_names(
    elements: list[dict], borrowed: dict[int, dict], screen_area: int
) -> tuple[dict[int, str], set[int]]:
    """Names for nameless toggles, and the duplicate rows they replace.

    A nameless switch or checkbox takes the name of its innermost named
    container. iOS reports a settings switch twice, as the named row and its
    unnamed knob; only the knob toggles, so the row of the same kind is
    dropped and the knob carries its name.
    """
    named = []
    for element in elements:
        if not _visible(element) or (bounds := _parse_bounds(element)) is None:
            continue
        if screen_area and _area(bounds) > screen_area * _MAX_OWNER_SCREEN_SHARE:
            continue
        texts = (borrowed.get(id(element)) or {}).get("texts") or []
        if name := _display_name(element) or ", ".join(texts):
            named.append((_area(bounds), bounds, element, name))
    names: dict[int, str] = {}
    dropped: set[int] = set()
    for element in elements:
        kind = _kind(element)
        if kind not in _TOGGLE_KINDS or _display_name(element):
            continue
        if (bounds := _parse_bounds(element)) is None or not _offerable(element):
            continue
        owner = min(
            (
                item
                for item in named
                if item[2] is not element
                and _inside(bounds, item[1], _KNOB_SLACK)
                and _contains(item[2], element)
            ),
            key=lambda item: item[0],
            default=None,
        )
        if owner is None:
            if any(
                _parse_bounds(row) in (None, (0, 0, 0, 0))
                for row in _ancestors(element, elements)
            ):
                # iOS keeps the knob of a row scrolled off screen at a stale spot.
                dropped.add(id(element))
            elif label := _label_beside(element, bounds, elements):
                names[id(element)] = label  # A switch next to its unlinked label.
            continue
        names[id(element)] = owner[3]
        if _kind(owner[2]) == kind:
            # The row itself, and its title button, would repeat the knob.
            dropped.add(id(owner[2]))
            for other in elements:
                if (
                    other is not element
                    and _contains(owner[2], other)
                    and type(owner[2].get("descendants")) is int
                    and _display_name(other) == owner[3]
                ):
                    dropped.add(id(other))
    return names, dropped


_LIST_CLASSES = {"CollectionView", "Table", "ScrollView"}
_BAR_CLASSES = {"NavigationBar", "Toolbar", "TabBar"}


def _sheet_frame(
    sheet: dict, bar: dict, elements: list[dict]
) -> tuple[int, int, int, int] | None:
    """The visible frame of a sheet: the largest on-screen element holding its
    navigation bar below the full-window container (a detent sheet leaves the
    page above it reachable)."""
    root = _parse_bounds(sheet)
    frames = [
        box
        for e in elements
        if e is not sheet
        and _contains(sheet, e)
        and _contains(e, bar)
        and (box := _parse_bounds(e)) is not None
        and box[0] >= 0
        and box[1] >= 0
        and box != root
    ]
    return max(frames, key=_area) if frames else root


# Return keys worth offering; the rest of the keyboard is typed by the delegate.
_SUBMIT_KEYS = {
    "search",
    "go",
    "done",
    "return",
    "send",
    "next",
    "join",
    "continue",
    "route",
}


def _parents(elements: list[dict]) -> dict[int, dict | None]:
    """Each element's parent, from the pre-order indices and descendant counts."""
    parents: dict[int, dict | None] = {}
    stack: list[tuple[dict, int]] = []
    for element in elements:
        index = element.get("index")
        while stack and not (type(index) is int and index <= stack[-1][1]):
            stack.pop()
        parents[id(element)] = stack[-1][0] if stack else None
        size = element.get("descendants")
        if type(index) is int and type(size) is int:
            stack.append((element, index + size))
    return parents


def _hidden_ios_elements(
    elements: list[dict], screen_area: int
) -> tuple[set[int], set[int]]:
    """iOS elements a tap cannot reach, and the keyboard keys to keep.

    The tree keeps everything below a sheet, a floating bar or the keyboard,
    with nothing marking it as covered. Later elements are drawn on top, so:
    a sheet (a later part of a window with its own navigation bar) hides the
    earlier parts it covers, an option whose tap point lies on a later
    option is covered by it, and the keyboard is typed by the delegate except
    for its return key.
    """
    nested = [e for e in elements if type(e.get("descendants")) is int]
    if not nested or not any(_is_ios(e) for e in elements):
        return set(), set()
    parents = _parents(elements)
    hidden: set[int] = set()
    submit_keys: set[int] = set()

    def centre(element: dict) -> tuple[int, int] | None:
        box = _parse_bounds(element)
        if box is None or box[2] <= box[0] or box[3] <= box[1]:
            return None
        return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2

    def within(point: tuple[int, int] | None, box: tuple[int, ...] | None) -> bool:
        return (
            point is not None
            and box is not None
            and box[0] <= point[0] < box[2]
            and box[1] <= point[1] < box[3]
        )

    for window in (e for e in elements if _short_class(e) == "Window"):
        children = [e for e in elements if parents.get(id(e)) is window]
        # A sheet hides the earlier parts of its window that it covers.
        for later, sheet in enumerate(children):
            bar = next(
                (
                    e
                    for e in elements
                    if _short_class(e) == "NavigationBar" and _contains(sheet, e)
                ),
                None,
            )
            if bar is None:
                continue
            cover = _sheet_frame(sheet, bar, elements)
            for below in children[:later]:
                hidden |= {
                    id(e)
                    for e in elements
                    if (e is below or _contains(below, e)) and within(centre(e), cover)
                }
    # The keys are typed by the delegate, except the return key. The keyboard
    # covers whatever lies behind it in any window; an input accessory bar
    # above the keys (a chat composer) stays.
    for keyboard in (e for e in elements if _short_class(e) == "Keyboard"):
        board = _parse_bounds(keyboard)
        if board is None:
            continue
        own_window = next(
            (
                w
                for w in elements
                if _short_class(w) == "Window" and _contains(w, keyboard)
            ),
            None,
        )
        for element in elements:
            if _short_class(element) == "Window" or _contains(element, keyboard):
                continue  # The keyboard's own ancestors are not covered by it.
            point = centre(element)
            if not (
                element is keyboard
                or _contains(keyboard, element)
                or (
                    point is not None
                    and point[1] >= board[1]
                    and board[0] <= point[0] < board[2]
                )
            ):
                continue
            names = {
                _display_name(element).strip().lower(),
                str(element.get("resourceId") or "").strip().lower(),
            }
            on_keyboard = own_window is not None and _contains(own_window, element)
            if on_keyboard and names & _SUBMIT_KEYS:
                submit_keys.add(id(element))
            else:
                hidden.add(id(element))
    # Bars are drawn over the content scrolling under them, whatever the order.
    bars = [e for e in elements if _short_class(e) in _BAR_CLASSES]
    on_bar = {id(e) for e in elements for bar in bars if e is bar or _contains(bar, e)}
    lists = [e for e in elements if _short_class(e) in _LIST_CLASSES]

    def in_list(element: dict) -> dict | None:
        """The innermost list holding ``element``."""
        holders = [
            lst for lst in lists if lst is not element and _contains(lst, element)
        ]
        return holders[-1] if holders else None

    # A later list drawn over an earlier one (search suggestions over the
    # page) hides it; a list nested in another is part of it.
    for position, lower in enumerate(lists):
        for upper in lists[position + 1 :]:
            if _contains(lower, upper) or _contains(upper, lower):
                continue
            cover = _parse_bounds(upper)
            hidden |= {
                id(e)
                for e in elements
                if _contains(lower, e)
                and id(e) not in on_bar
                and within(centre(e), cover)
            }
    # An option whose tap point lies on a later, separate option is covered.
    offered = [e for e in elements if _offerable(e) and id(e) not in hidden]
    small = [
        e
        for e in offered
        if (box := _parse_bounds(e)) is not None
        and (not screen_area or _area(box) < screen_area * _MAX_OWNER_SCREEN_SHARE)
    ]

    def window_part(element: dict) -> dict | None:
        """The top-level child of the window holding ``element``."""
        node = element
        while (parent := parents.get(id(node))) is not None:
            if _short_class(parent) == "Window":
                return node
            node = parent
        return None

    for element in offered:
        point, holder = centre(element), in_list(element)
        part = window_part(element)
        for cover in small:
            # Bars sit over their own page's content, but not over an overlay
            # presented later (an action sheet over the tab bar).
            if id(element) in on_bar and window_part(cover) is part:
                continue
            # A row scrolled under something covers nothing outside its list.
            cover_list = in_list(cover)
            if cover_list is not None and cover_list is not holder:
                continue
            if (
                cover["index"] > element["index"]
                and not _contains(element, cover)
                and not _contains(cover, element)
                and _parse_bounds(cover) != _parse_bounds(element)
                and within(point, _parse_bounds(cover))
            ):
                hidden.add(id(element))
                break
    return hidden - submit_keys, submit_keys


def _option_name(
    element: dict,
    borrowed: dict[int, dict],
    toggle_names: dict[int, str],
    field_labels: dict[int, str],
) -> str:
    texts = (borrowed.get(id(element)) or {}).get("texts") or []
    facts = element.get("details") or {}
    if _kind(element) in _TEXT_KINDS:  # A field is known by its name, not its contents.
        name = (
            facts.get("description")
            or facts.get("hint")
            or field_labels.get(id(element))
        )
        if name:
            return str(name).strip().lower()
    return (
        (_display_name(element) or toggle_names.get(id(element)) or ", ".join(texts))
        .strip()
        .lower()
    )


def _same_named_wrappers(
    elements: list[dict],
    borrowed: dict[int, dict],
    toggle_names: dict[int, str],
    field_labels: dict[int, str] | None = None,
) -> set[int]:
    """Rows whose nested button or switch is an option with the same name.

    The inner control is offered instead: a settings row and its switch, or an
    iOS cell and the button filling it, would otherwise be two equal options.
    """
    offered = [element for element in elements if _offerable(element)]
    names = {
        id(e): _option_name(e, borrowed, toggle_names, field_labels or {})
        for e in offered
    }
    wrappers = set()
    for container in offered:
        if (
            not (name := names[id(container)])
            or (box := _parse_bounds(container)) is None
        ):
            continue
        for inner in offered:
            if (
                inner is not container
                # A row may open a page while only its switch toggles, so a
                # toggle replaces only its iOS twin of the same kind.
                and (
                    _kind(inner) not in _TOGGLE_KINDS
                    or _kind(container) == _kind(inner)
                )
                and names[id(inner)] == name
                and (inner_box := _parse_bounds(inner)) is not None
                and inner_box != box
                and _inside(inner_box, box, _KNOB_SLACK)
                and _contains(container, inner)
            ):
                wrappers.add(id(container))
                break
    return wrappers


def _ancestors(element: dict, elements: list[dict]) -> list[dict]:
    """Elements that the tree says contain ``element``."""
    return [
        other
        for other in elements
        if other is not element
        and type(other.get("descendants")) is int
        and _contains(other, element)
    ]


def _label_beside(element: dict, bounds: tuple[int, ...], elements: list[dict]) -> str:
    """The text just left of a control, on the same line."""
    center_y = (bounds[1] + bounds[3]) // 2
    best = None
    for other in elements:
        if other is element or not _visible(other) or _kind(other):
            continue
        if not (name := _display_name(other)) or (box := _parse_bounds(other)) is None:
            continue
        if box[1] <= center_y <= box[3] and box[2] <= bounds[0] + _KNOB_SLACK:
            if best is None or box[2] > best[0]:
                best = (box[2], name)
    return best[1] if best else ""


def _candidate_label(
    element: dict,
    borrowed: dict[int, dict],
    toggle_names: dict[int, str],
    focused: int | None,
    center_y: int,
    screen_height: int | None,
    field_labels: dict[int, str] | None = None,
    submit_keys: set[int] | None = None,
) -> str:
    """Describe an element: its kind, its name, and its state and position."""
    field_labels = field_labels or {}
    inner = borrowed.get(id(element)) or {"texts": [], "widget": None}
    facts = element.get("details") or {}
    kind, state = _kind(element), _check_state(element)
    if submit_keys and id(element) in submit_keys:
        kind = "keyboard key"
    resource = str(element.get("resourceId") or "").rsplit("/", 1)[-1]
    if not (name := _display_name(element) or toggle_names.get(id(element), "")):
        if kind in _TEXT_KINDS:
            name = _own_text(element)
        if not name and (texts := inner["texts"]):
            joined = ", ".join(texts)
            # An id says more than a bare badge or counter ("menu_cart: 3").
            badge = not any(char.isalpha() for char in joined)
            name = f"{resource}: {joined}" if resource and badge else joined
    if not name:
        short = _short_class(element)
        unnamed = "unnamed" if kind else f"unnamed {short}" if short else ""
        name = resource or unnamed or str(element["index"])
    if not kind and facts.get("editable"):
        kind = "text field"
    if inner["widget"] is not None:
        # A row whose switch or checkbox is a separate element toggles it.
        inner_kind, inner_state = inner["widget"]
        kind, state = kind or inner_kind, state or inner_state
    name, contents = _name_and_contents(
        element, kind, name, field_labels.get(id(element), "")
    )
    notes = [note for note in (contents, state) if note]
    extra = str(facts.get("stateDescription") or "")
    if extra and not state and extra not in name:
        notes.append(extra)
    if facts.get("password"):
        notes.append("password")
    if facts.get("selected"):
        notes.append("selected")
    if facts.get("focused") or (focused is not None and id(element) == focused):
        notes.append("focused")
    if screen_height:
        if center_y < screen_height * _EDGE_SHARE:
            notes.append("top of screen")
        elif center_y > screen_height * (1 - _EDGE_SHARE):
            notes.append("bottom of screen")
    label = f'{kind} "{name}"' if kind else name
    return f"{label} ({', '.join(notes)})" if notes else label


def should_resize_model_screenshot(state_provider: Any) -> bool:
    """True when screenshots attached to LLM messages must be resized (with
    the labeled grid) into the coordinate space the provider declared.

    Falls back to the legacy ``requires_coordinate_tools`` contract: injected
    screenshot-only providers that predate ``resize_model_screenshot`` are
    still treated as screenshot-coordinate mode by the tool registry, prompts,
    and validation, so they must keep receiving resized screenshots too.
    """
    return bool(
        getattr(state_provider, "resize_model_screenshot", False)
        or getattr(state_provider, "requires_coordinate_tools", False)
    )


def resize_model_screenshot_with_grid(state_provider: Any, screenshot: bytes) -> bytes:
    """Resize a screenshot to the exact model-facing dims the provider declared.

    The provider stores ``model_screenshot_width/height`` (resolved from the
    active vision model's effective image size) on each ``get_state``. Resizing
    to those exact dims keeps the image the model grounds on equal to the
    declared coordinate space. Falls back to the legacy max-side resize when no
    dims are available (e.g. injected screenshot-only providers)."""
    width = getattr(state_provider, "model_screenshot_width", None)
    height = getattr(state_provider, "model_screenshot_height", None)
    # A pixel grid contradicts a normalized coordinate contract.
    divisions = 10 if getattr(state_provider, "model_screenshot_grid", True) else 0
    if width and height:
        return resize_image_to_dimensions_with_grid(
            screenshot, width, height, divisions=divisions
        )
    return resize_image_to_max_side_with_grid(screenshot, divisions=divisions)


async def resize_model_screenshot_async(
    state_provider: Any, screenshot: bytes
) -> bytes:
    """Resize when required, snapshotting model geometry before offloading."""
    if not should_resize_model_screenshot(state_provider):
        return screenshot

    geometry = SimpleNamespace(
        model_screenshot_width=getattr(state_provider, "model_screenshot_width", None),
        model_screenshot_height=getattr(
            state_provider, "model_screenshot_height", None
        ),
        model_screenshot_grid=getattr(state_provider, "model_screenshot_grid", True),
    )
    return await asyncio.to_thread(
        resize_model_screenshot_with_grid, geometry, screenshot
    )


class AndroidStateProvider(StateProvider):
    """Fetches state from an Android device via ``driver.get_ui_tree()``.

    Includes retry logic with exponential backoff and mid-retry recovery
    (accessibility service restart) for robustness against intermittent
    Portal/a11y failures.
    """

    supported = {"element_index", "convert_point"}

    def __init__(
        self,
        driver: "DeviceDriver",
        tree_filter: "TreeFilter",
        tree_formatter: "TreeFormatter",
        use_normalized: bool = False,
        stealth: bool = False,
        ui_cls: "type[UIState] | None" = None,
        vision_enabled: bool = False,
        vision_resize_policy: Any = None,
    ) -> None:
        super().__init__(driver)
        self.tree_filter = tree_filter
        self.tree_formatter = tree_formatter
        self.use_normalized = use_normalized
        self._ui_cls = ui_cls or (StealthUIState if stealth else UIState)
        self.vision_enabled = vision_enabled
        # Resolves the exact screenshot dims the active vision model grounds on
        # (duck-typed: needs ``effective_dims(w, h)``). None → legacy 2048 cap.
        self.vision_resize_policy = vision_resize_policy
        # Exact model-facing screenshot dims for the current state (set per
        # get_state); read by resize_model_screenshot_with_grid.
        self.model_screenshot_width: Optional[int] = None
        self.model_screenshot_height: Optional[int] = None
        # Model coordinates reach device pixels either 1:1, through the vision
        # contract's scaling, or through normalized [0-1000] conversion.
        self.screenshot_matches_input_coords = True
        # Normalized [0-1000] coordinates are scale-invariant, so the
        # deterministic-space contract is only needed for pixel coordinates.
        # Re-evaluated per get_state: the flag must describe the CURRENT
        # state's contract — if screen bounds are unavailable for a state,
        # resizing that step's screenshot would desynchronize the model's
        # space from convert_point.
        self._vision_contract_intent = vision_enabled and not use_normalized
        self.resize_model_screenshot = self._vision_contract_intent
        # Hybrid vision exposes click_at when a coordinate contract is
        # expected, but individual states can still arrive with missing or
        # zero screen bounds. Refuse coordinate actions for those snapshots
        # instead of mapping model coordinates against a wrong screen size.
        self.requires_active_contract_for_coords = vision_enabled

    def _portal_recovery(self) -> Optional[Callable[[], Awaitable[None]]]:
        """Return Portal recovery, or None when there is no ADB device."""
        from mobilerun_core_local.driver.android import AndroidDriver

        from mobilerun.tools.driver.secret_safe import SecretSafeDriver

        # The redaction wrapper is transparent for the concrete-driver check.
        driver = self.driver
        if isinstance(driver, SecretSafeDriver):
            driver = driver.inner
        if isinstance(driver, AndroidDriver) and driver.device is not None:
            return self._recover_portal
        return None

    async def _recover_portal(self) -> None:
        """Restart Portal's accessibility service, TCP server and keyboard."""
        device = self.driver.device

        from mobilerun_core_local.driver.android.portal import (
            PORTAL_PACKAGE_NAME,
            portal_a11y_service,
            portal_content_uri,
        )

        a11y = portal_a11y_service(PORTAL_PACKAGE_NAME)

        # 1. Restart accessibility service
        logger.debug("Restarting Portal accessibility service...")
        await device.shell("settings put secure accessibility_enabled 0")
        await asyncio.sleep(0.5)
        await device.shell(f"settings put secure enabled_accessibility_services {a11y}")
        await device.shell("settings put secure accessibility_enabled 1")

        # 2. Restart TCP socket server if it was in use
        portal = self.driver.portal
        if portal is not None and portal.tcp_available:
            logger.debug("Restarting Portal TCP socket server...")
            toggle_uri = portal_content_uri(PORTAL_PACKAGE_NAME, "toggle_socket_server")
            try:
                await device.shell(
                    f"content insert --uri {toggle_uri} --bind enabled:b:false"
                )
                await asyncio.sleep(0.3)
                await device.shell(
                    f"content insert --uri {toggle_uri} --bind enabled:b:true"
                )
                # Re-fetch auth token — server restart may rotate it
                new_token = await portal._fetch_auth_token()
                if new_token:
                    portal._auth_token = new_token
                    logger.debug("Auth token refreshed after TCP server restart")
            except Exception as e:
                logger.debug(f"TCP server restart failed: {e}")

        # 3. A force-stopped Portal also loses the keyboard; switch back to it
        if getattr(self.driver, "_portal_keyboard_available", False):
            from mobilerun_core_local.driver.android.portal import setup_keyboard

            try:
                await setup_keyboard(device)
            except Exception as e:
                logger.debug(f"Portal keyboard restore failed: {e}")

        await asyncio.sleep(1.5)

    async def get_state(self) -> UIState:
        combined_data = await fetch_state_with_retry(
            fetch=self.driver.get_ui_tree,
            recovery=self._portal_recovery(),
        )
        return await self._state_from_data_async(combined_data)

    async def get_final_state(self) -> UIState:
        """Finalization must not restart accessibility after a failed read."""
        combined_data = await self.driver.get_ui_tree()
        return await self._state_from_data_async(combined_data)

    async def _state_from_data_async(self, combined_data: dict) -> UIState:
        # Workers may outlive cancellation. Only their per-call copies may change.
        processor = copy(self)
        processor.tree_filter = deepcopy(self.tree_filter)
        processor.tree_formatter = deepcopy(self.tree_formatter)
        processor.vision_resize_policy = deepcopy(self.vision_resize_policy)

        def process() -> tuple[UIState, bool]:
            state = processor._state_from_data(combined_data)
            return state, processor.resize_model_screenshot

        state, resize = await asyncio.to_thread(process)
        # Publish together on the loop only when processing completed successfully.
        self.resize_model_screenshot = resize
        self.model_screenshot_width = state.model_screenshot_width
        self.model_screenshot_height = state.model_screenshot_height
        self.tree_formatter.screen_width = state.screen_width
        self.tree_formatter.screen_height = state.screen_height
        self.tree_formatter.use_normalized = state.use_normalized
        self.tree_formatter.display_scale_x = state.coordinate_scale_x
        self.tree_formatter.display_scale_y = state.coordinate_scale_y
        return state

    def _state_from_data(self, combined_data: dict) -> UIState:

        device_context = combined_data["device_context"]
        screen_bounds = device_context.get("screen_bounds", {})
        screen_width = screen_bounds.get("width")
        screen_height = screen_bounds.get("height")

        filtered = self.tree_filter.filter(combined_data["a11y_tree"], device_context)

        coordinate_scale_x = 1.0
        coordinate_scale_y = 1.0
        display_width = None
        display_height = None
        if self._vision_contract_intent and screen_width and screen_height:
            if self.vision_resize_policy is not None:
                display_width, display_height = (
                    self.vision_resize_policy.effective_dims(
                        screen_width, screen_height
                    )
                )
            else:
                display_width, display_height = fit_dimensions_to_max_side(
                    screen_width, screen_height
                )
            coordinate_scale_x = screen_width / display_width
            coordinate_scale_y = screen_height / display_height
        if self._vision_contract_intent:
            # Keep agent-side resizing paired with the contract this state
            # actually declares (screen bounds can be missing for a state).
            self.resize_model_screenshot = bool(display_width and display_height)
        # Exact dims the model-facing screenshot must match for this state.
        self.model_screenshot_width = display_width
        self.model_screenshot_height = display_height

        self.tree_formatter.screen_width = screen_width
        self.tree_formatter.screen_height = screen_height
        self.tree_formatter.use_normalized = self.use_normalized
        # Model-facing bounds text is emitted in the display space the model
        # actually sees; tap bounds in ``elements`` stay native device pixels.
        self.tree_formatter.display_scale_x = coordinate_scale_x
        self.tree_formatter.display_scale_y = coordinate_scale_y

        formatted_text, focused_text, elements, phone_state = (
            self.tree_formatter.format(filtered, combined_data["phone_state"])
        )

        if (
            self.use_normalized
            and self.vision_enabled
            and screen_width
            and screen_height
        ):
            formatted_text += f"\n\n{NORMALIZED_STATE_NOTE}"
        if display_width and display_height:
            scaled_note = (
                f" — the {screen_width}x{screen_height} device screen scaled down"
                if (display_width, display_height) != (screen_width, screen_height)
                else ""
            )
            formatted_text += (
                f"\n\nAll coordinates in this device state (element bounds above, "
                f"and any x/y you provide to coordinate actions) are in a "
                f"{display_width}x{display_height} coordinate space{scaled_note}. "
                f"When a screenshot is provided it matches this "
                f"{display_width}x{display_height} space and carries a labeled "
                f"coordinate grid for reference."
            )

        return self._ui_cls(
            elements=elements,
            formatted_text=formatted_text,
            focused_text=focused_text,
            phone_state=phone_state,
            screen_width=screen_width,
            screen_height=screen_height,
            use_normalized=self.use_normalized,
            coordinate_scale_x=coordinate_scale_x,
            coordinate_scale_y=coordinate_scale_y,
            coordinate_contract_active=bool(display_width and display_height)
            or bool(self.use_normalized and screen_width and screen_height),
            model_screenshot_width=display_width,
            model_screenshot_height=display_height,
        )
