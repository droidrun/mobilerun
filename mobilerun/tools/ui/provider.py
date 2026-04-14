"""StateProvider — orchestrates fetching and parsing device state.

Fetches raw data from a ``DeviceDriver``, applies tree filters/formatters,
and produces a ``UIState`` snapshot.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from droidrun.tools.driver.base import DeviceDisconnectedError
from droidrun.tools.ui.state import UIState
from droidrun.tools.ui.stealth_state import StealthUIState

if TYPE_CHECKING:
    from droidrun.tools.driver.base import DeviceDriver
    from droidrun.tools.filters import TreeFilter
    from droidrun.tools.formatters import TreeFormatter

logger = logging.getLogger("droidrun")

# Retry schedule: delay in seconds after each failed attempt.
# Total wait across 7 attempts: 1+2+3+5+8+10 = 29s.
_RETRY_DELAYS = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
_MAX_RETRIES = 7

# After this many consecutive failures, run the recovery callback.
# With the schedule above, this fires after ~11s (1+2+3+5).
_RECOVERY_AFTER_ATTEMPT = 5


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
            consecutive failures (e.g. restart accessibility service).
        max_retries: Total number of attempts before giving up.
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

    def __init__(self, driver: "DeviceDriver") -> None:
        self.driver = driver

    async def get_state(self) -> UIState:
        raise NotImplementedError


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
    ) -> None:
        super().__init__(driver)
        self.tree_filter = tree_filter
        self.tree_formatter = tree_formatter
        self.use_normalized = use_normalized
        self._ui_cls = ui_cls or (StealthUIState if stealth else UIState)

    async def _recover_portal(self) -> None:
        """Restart Portal's accessibility service and TCP socket server."""
        from droidrun.tools.driver.android import AndroidDriver

        if not isinstance(self.driver, AndroidDriver):
            return
        device = self.driver.device
        if device is None:
            return

        from droidrun.portal import A11Y_SERVICE_NAME

        # 1. Restart accessibility service
        logger.debug("Restarting Portal accessibility service...")
        await device.shell("settings put secure accessibility_enabled 0")
        await asyncio.sleep(0.5)
        await device.shell(
            f"settings put secure enabled_accessibility_services {A11Y_SERVICE_NAME}"
        )
        await device.shell("settings put secure accessibility_enabled 1")

        # 2. Restart TCP socket server if it was in use
        portal = self.driver.portal
        if portal is not None and portal.tcp_available:
            logger.debug("Restarting Portal TCP socket server...")
            try:
                await device.shell(
                    "content insert --uri content://com.droidrun.portal/toggle_socket_server --bind enabled:b:false"
                )
                await asyncio.sleep(0.3)
                await device.shell(
                    "content insert --uri content://com.droidrun.portal/toggle_socket_server --bind enabled:b:true"
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
        )

        device_context = combined_data["device_context"]
        screen_bounds = device_context.get("screen_bounds", {})
        screen_width = screen_bounds.get("width")
        screen_height = screen_bounds.get("height")

        filtered = self.tree_filter.filter(combined_data["a11y_tree"], device_context)

        self.tree_formatter.screen_width = screen_width
        self.tree_formatter.screen_height = screen_height
        self.tree_formatter.use_normalized = self.use_normalized

        formatted_text, focused_text, elements, phone_state = (
            self.tree_formatter.format(filtered, combined_data["phone_state"])
        )

        return self._ui_cls(
            elements=elements,
            formatted_text=formatted_text,
            focused_text=focused_text,
            phone_state=phone_state,
            screen_width=screen_width,
            screen_height=screen_height,
            use_normalized=self.use_normalized,
        )
