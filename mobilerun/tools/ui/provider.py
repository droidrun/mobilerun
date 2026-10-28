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
        for element in UIState._collect_all(state.elements):
            index = element.get("index")
            if (
                type(index) is not int
                or index in seen
                or element.get("isEnabled") is False
                or element.get("isVisibleToUser") is False
            ):
                continue
            seen.add(index)
            bounds = element.get("bounds")
            if not isinstance(bounds, str):
                continue
            try:
                left, top, right, bottom = map(int, bounds.split(","))
                if right <= left or bottom <= top:
                    continue
                if state.screen_width and (right <= 0 or left >= state.screen_width):
                    continue
                if state.screen_height and (bottom <= 0 or top >= state.screen_height):
                    continue
                UIState._avoid_tap_blockers(
                    state, element, ((left + right) // 2, (top + bottom) // 2)
                )
            except (KeyError, TypeError, ValueError):
                continue
            label = element.get("text") or element.get("resourceId") or str(index)
            if element.get("isClickable", True):
                options.append(
                    ActionCandidate(ToolCall("click", {"index": index}), f"Click {label}")
                )
            if element.get("isLongClickable", False):
                options.append(
                    ActionCandidate(ToolCall("long_press", {"index": index}), f"Long press {label}")
                )
        return tuple(options)

    async def build_action_candidates(
        self,
        ctx: ActionContext,
        tools: dict[str, Any],
        *,
        include_device_actions: bool = False,
    ) -> tuple[ActionCandidate, ...]:
        """Keep provider overrides and add only enabled, supported device actions."""
        from mobilerun.agent.fast_agent.xml_parser import ToolCall
        from mobilerun.agent.selectors.base import ActionCandidate

        options = list(self.get_action_candidates(ctx.ui)) if ctx.ui is not None else []
        if not include_device_actions:
            return tuple(options)

        if "system_button" in tools:
            navigation = {"back", "home", "enter", "recent", "recents", "app_switch", "overview"}
            for button in sorted(getattr(self.driver, "supported_buttons", set()) & navigation):
                options.append(ActionCandidate(
                    ToolCall("system_button", {"button": button}),
                    f"Press {button}: " + {
                        "back": "go back or dismiss the keyboard/dialog",
                        "home": "go to the home screen without closing apps",
                        "enter": "submit the focused field",
                    }.get(button, "show recent apps"),
                ))

        if "swipe" in tools:
            options.extend(self._swipe_candidates(ctx.ui))
        if "wait" in tools:
            options.append(ActionCandidate(
                ToolCall("wait", {"duration": 1.0}),
                "Wait 1 second for a loading screen, animation, or UI update",
            ))
        if "open_app" in tools and "get_apps" in getattr(self.driver, "supported", set()):
            options.extend(await self._app_candidates(ctx, tools["open_app"]))

        # A provider may already supply some device actions. Preserve its order and
        # avoid duplicate choices without touching the executable parameters.
        unique: list[ActionCandidate] = []
        for option in options:
            if option.action.name in tools and not any(option.action == item.action for item in unique):
                unique.append(option)
        return tuple(unique)

    def _swipe_candidates(self, state: UIState | None) -> list[ActionCandidate]:
        from mobilerun.agent.fast_agent.xml_parser import ToolCall
        from mobilerun.agent.selectors.base import ActionCandidate

        if state is None or not state.screen_width or not state.screen_height:
            return []
        if (
            getattr(self, "requires_active_contract_for_coords", False)
            and not state.coordinate_contract_active
        ):
            return []
        if state.use_normalized:
            width = height = 1000
        elif getattr(self, "requires_coordinate_tools", False):
            width, height = state.screen_width, state.screen_height
        else:
            try:
                width = state.screen_width / state.coordinate_scale_x
                height = state.screen_height / state.coordinate_scale_y
            except (TypeError, ValueError, ZeroDivisionError):
                return []
        if not all(math.isfinite(value) and value >= 4 for value in (width, height)):
            return []

        def point(x: float, y: float) -> list[int]:
            return [round((width - 1) * x), round((height - 1) * y)]

        return [ActionCandidate(
            ToolCall("swipe", {"coordinate": point(*start), "coordinate2": point(*end), "duration": 0.5}),
            description,
        ) for start, end, description in (
            ((0.5, 0.8), (0.5, 0.2), "Swipe up to scroll down or reveal apps from the home screen"),
            ((0.5, 0.2), (0.5, 0.8), "Swipe down to scroll up"),
            ((0.8, 0.5), (0.2, 0.5), "Swipe left to the next page or panel"),
            ((0.2, 0.5), (0.8, 0.5), "Swipe right to the previous page or panel"),
        )]

    async def _app_candidates(
        self, ctx: ActionContext, signature: dict[str, Any],
    ) -> list[ActionCandidate]:
        from mobilerun.agent.action_context import ActionCandidateApp
        from mobilerun.agent.fast_agent.xml_parser import ToolCall
        from mobilerun.agent.selectors.base import ActionCandidate

        params = signature.get("parameters", {})
        parameter = next((name for name in ("app_id", "bundle_id", "text") if name in params), None)
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
                identifier = next((app.get(key) for key in (
                    "package", "package_name", "bundle_id", "app_id",
                ) if isinstance(app.get(key), str) and app[key].strip()), None)
                if identifier:
                    label = app.get("label") or app.get("name") or identifier
                    installed[identifier] = ActionCandidateApp(identifier=identifier, label=str(label))
            ctx.action_candidate_apps = tuple(sorted(
                installed.values(), key=lambda app: (app.label.casefold(), app.identifier),
            ))

        current_app = ctx.ui.phone_state.get("packageName") if ctx.ui is not None else None
        return [ActionCandidate(
            ToolCall("open_app", {parameter: app.identifier}),
            f"Open app {app.label} ({app.identifier})",
        ) for app in ctx.action_candidate_apps if app.identifier != current_app]


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

        if isinstance(self.driver, AndroidDriver) and self.driver.device is not None:
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
