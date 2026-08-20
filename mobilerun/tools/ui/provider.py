"""StateProvider — orchestrates fetching and parsing device state.

Fetches raw data from a ``DeviceDriver``, applies tree filters/formatters,
and produces a ``UIState`` snapshot.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from mobilerun_core_local.driver.base import DeviceDisconnectedError

from mobilerun.tools.helpers.images import (
    fit_dimensions_to_max_side,
    resize_image_to_dimensions_with_grid,
    resize_image_to_max_side_with_grid,
)
from mobilerun.tools.ui.state import UIState
from mobilerun.tools.ui.stealth_state import StealthUIState

if TYPE_CHECKING:
    from mobilerun_core_local.driver.base import DeviceDriver

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

# Max number of re-fetches used to confirm the a11y tree has stopped
# changing. Bounds the extra latency ``wait_for_stable_ui`` can add per
# get_state() call to roughly max_stability_checks * wait_for_stable_ui.
_MAX_STABILITY_CHECKS = 3


async def _wait_for_stable_ui_tree(
    fetch: Callable[[], Awaitable[Dict[str, Any]]],
    initial: Dict[str, Any],
    wait_for_stable_ui: float,
    max_checks: int,
) -> Dict[str, Any]:
    """Re-fetch state until the a11y tree stops changing between fetches.

    A well-formed dump can still be stale: e.g. a WebView's Chromium
    accessibility bridge lagging behind a client-side route change reports
    the previous screen with no error and all required keys present, so
    ``fetch_state_with_retry``'s error/missing-key checks accept it
    immediately. Polling until two consecutive fetches agree catches that
    case instead of handing the agent a screen it already navigated away
    from.
    """
    current = initial
    for _ in range(max_checks):
        await asyncio.sleep(wait_for_stable_ui)
        try:
            next_data = await fetch()
        except DeviceDisconnectedError:
            raise
        except Exception as e:
            logger.debug(f"UI-stability re-fetch failed, using last state: {e}")
            break
        if "error" in next_data or "a11y_tree" not in next_data:
            break
        if next_data["a11y_tree"] == current["a11y_tree"]:
            return next_data
        current = next_data
    return current


async def fetch_state_with_retry(
    fetch: Callable[[], Awaitable[Dict[str, Any]]],
    recovery: Optional[Callable[[], Awaitable[None]]] = None,
    max_retries: int = _MAX_RETRIES,
    retry_delays: Optional[List[float]] = None,
    recovery_after: int = _RECOVERY_AFTER_ATTEMPT,
    wait_for_stable_ui: float = 0.0,
    max_stability_checks: int = _MAX_STABILITY_CHECKS,
) -> Dict[str, Any]:
    """Fetch raw device state with retries, backoff, and mid-retry recovery.

    Args:
        fetch: Async callable that returns the raw state dict from Portal.
        recovery: Optional async callable invoked once after *recovery_after*
            consecutive failures (e.g. restart accessibility service).
        max_retries: Total number of attempts before giving up.
        retry_delays: Per-attempt delays. If shorter than max_retries - 1,
            the last value is reused for remaining delays.
        recovery_after: Trigger *recovery* after this many failures.
        wait_for_stable_ui: Seconds to wait between re-fetches when
            confirming the a11y tree has stopped changing. 0 disables the
            stability check (the first well-formed dump is trusted as-is).
        max_stability_checks: Cap on re-fetches while waiting for the tree
            to stabilize, so a screen that never settles (e.g. an animation
            loop) doesn't hang get_state() indefinitely.

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

    is_debug = logger.isEnabledFor(logging.DEBUG)

    for attempt in range(max_retries):
        try:
            logger.debug(f"Getting state (attempt {attempt + 1}/{max_retries})")

            t0 = time.monotonic() if is_debug else 0
            combined_data = await fetch()

            if is_debug:
                elapsed = (time.monotonic() - t0) * 1000
                logger.debug(f"State fetched in {elapsed:.0f}ms")

            if "error" in combined_data:
                raise Exception(f"Portal returned error: {combined_data}")

            required_keys = ["a11y_tree", "phone_state", "device_context"]
            missing = [k for k in required_keys if k not in combined_data]
            if missing:
                raise Exception(f"Missing data in state: {', '.join(missing)}")

            if wait_for_stable_ui > 0:
                combined_data = await _wait_for_stable_ui_tree(
                    fetch, combined_data, wait_for_stable_ui, max_stability_checks
                )

            return combined_data

        except DeviceDisconnectedError:
            raise
        except Exception as e:
            last_error = e
            is_last = attempt >= max_retries - 1
            delay = delays[attempt] if attempt < len(delays) else delays[-1]

            err_desc = str(e) or type(e).__name__
            suffix = f" (retrying in {delay:.0f}s)" if not is_last else ""
            logger.warning(
                f"get_state attempt {attempt + 1} failed: {err_desc}{suffix}"
            )

            # Mid-retry recovery: restart the a11y service once
            if (
                not recovery_attempted
                and recovery is not None
                and attempt + 1 >= recovery_after
                and not is_last
            ):
                recovery_attempted = True
                logger.info("State retrieval failing, attempting recovery...")
                try:
                    await recovery()
                    logger.info("Recovery action completed")
                except Exception as rec_err:
                    logger.warning(f"Recovery action failed: {rec_err}")

            if not is_last:
                await asyncio.sleep(delay)

    last_desc = str(last_error) or type(last_error).__name__
    error_msg = f"Failed to get state after {max_retries} attempts: {last_desc}"
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
    if width and height:
        return resize_image_to_dimensions_with_grid(screenshot, width, height)
    return resize_image_to_max_side_with_grid(screenshot)


class AndroidStateProvider(StateProvider):
    """Fetches state from an Android device via ``driver.get_ui_tree()``.

    Includes retry logic with exponential backoff and mid-retry recovery
    (accessibility service restart) for robustness against intermittent
    Portal/a11y failures, plus an optional post-fetch stability check
    (``wait_for_stable_ui``) that re-polls until the a11y tree stops
    changing, guarding against well-formed but stale dumps (e.g. a WebView
    a11y bridge lagging behind a client-side route change).
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
        wait_for_stable_ui: float = 0.0,
    ) -> None:
        super().__init__(driver)
        self.tree_filter = tree_filter
        self.tree_formatter = tree_formatter
        self.use_normalized = use_normalized
        self.wait_for_stable_ui = wait_for_stable_ui
        self._ui_cls = ui_cls or (StealthUIState if stealth else UIState)
        self.vision_enabled = vision_enabled
        # Resolves the exact screenshot dims the active vision model grounds on
        # (duck-typed: needs ``effective_dims(w, h)``). None → legacy 2048 cap.
        self.vision_resize_policy = vision_resize_policy
        # Exact model-facing screenshot dims for the current state (set per
        # get_state); read by resize_model_screenshot_with_grid.
        self.model_screenshot_width: Optional[int] = None
        self.model_screenshot_height: Optional[int] = None
        # Android screenshots and input taps share device-pixel coordinates,
        # but only when not in normalized mode. ``use_normalized=True`` makes
        # ``UIState.convert_point`` treat inputs as [0-1000] normalized
        # coordinates, which is incompatible with picking coordinates off the
        # screenshot — keep click_at masked in that case.
        self.screenshot_matches_input_coords = not use_normalized
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
        # instead of treating model coordinates as native device pixels.
        self.requires_active_contract_for_coords = self._vision_contract_intent

    async def _recover_portal(self) -> None:
        """Restart Portal's accessibility service and TCP socket server."""
        from mobilerun_core_local.driver.android import AndroidDriver

        if not isinstance(self.driver, AndroidDriver):
            return
        device = self.driver.device
        if device is None:
            return

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

        await asyncio.sleep(1.5)

    async def get_state(self) -> UIState:
        combined_data = await fetch_state_with_retry(
            fetch=self.driver.get_ui_tree,
            recovery=self._recover_portal,
            wait_for_stable_ui=self.wait_for_stable_ui,
        )

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
            coordinate_contract_active=bool(display_width and display_height),
            model_screenshot_width=display_width,
            model_screenshot_height=display_height,
        )
