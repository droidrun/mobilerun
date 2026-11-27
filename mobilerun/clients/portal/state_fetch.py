"""Portal-specific state retrieval, retries, and driver recovery."""

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx
from mobilerun_core_local.driver.base import DeviceDisconnectedError

from mobilerun.clients.portal.errors_text import (
    ACCESSIBILITY_SERVICE_NOT_AVAILABLE as _A11Y_SERVICE_NOT_AVAILABLE,
)
from mobilerun.config.settings import get_settings
from mobilerun.errors import StateFetchFailed

logger = logging.getLogger("mobilerun")


def _is_timeout(error: BaseException) -> bool:
    """True for device-side read timeouts; exclude connect and pool timeouts."""
    return isinstance(error, (TimeoutError, httpx.ReadTimeout))


async def _fetch_with_notice(
    fetch: Callable[[], Awaitable[Dict[str, Any]]], notice: bool, notice_after: float
) -> Dict[str, Any]:
    """Await ``fetch``, logging once when it exceeds the configured notice delay."""
    if not notice:
        return await fetch()
    task = asyncio.ensure_future(fetch())
    try:
        done, _ = await asyncio.wait({task}, timeout=notice_after)
    except BaseException:
        task.cancel()
        raise
    if not done:
        logger.info("Waiting for the UI state from the device...")
    return await task


async def fetch_state_with_retry(
    fetch: Callable[[], Awaitable[Dict[str, Any]]],
    recovery: Optional[Callable[[], Awaitable[None]]] = None,
    max_retries: Optional[int] = None,
    retry_delays: Optional[List[float]] = None,
    recovery_after: Optional[int] = None,
) -> Dict[str, Any]:
    """Fetch Portal state with configured retries, backoff, and recovery."""
    configured = get_settings()
    max_retries = (
        max_retries
        if max_retries is not None
        else configured.MOBILERUN_STATE_MAX_RETRIES
    )
    recovery_after = (
        recovery_after
        if recovery_after is not None
        else configured.MOBILERUN_STATE_RECOVERY_AFTER_ATTEMPT
    )
    max_timeout_attempts = configured.MOBILERUN_STATE_MAX_TIMEOUT_ATTEMPTS
    slow_fetch_notice_s = configured.MOBILERUN_STATE_SLOW_FETCH_NOTICE_S
    delays = (
        retry_delays
        if retry_delays is not None
        else configured.MOBILERUN_STATE_RETRY_DELAYS
    )
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
                combined_data = await _fetch_with_notice(
                    fetch, not slow_notice_logged, slow_fetch_notice_s
                )
            finally:
                if time.monotonic() - t0 >= slow_fetch_notice_s:
                    slow_notice_logged = True
            if is_debug:
                elapsed = (time.monotonic() - t0) * 1000
                logger.debug(f"State fetched in {elapsed:.0f}ms")
            if "error" in combined_data:
                portal_error = combined_data["error"]
                if (
                    isinstance(portal_error, str)
                    and _A11Y_SERVICE_NOT_AVAILABLE in portal_error
                ):
                    raise Exception(_A11Y_SERVICE_NOT_AVAILABLE)
                raise Exception("Portal returned an error")
            required_keys = ["a11y_tree", "phone_state", "device_context"]
            missing = [key for key in required_keys if key not in combined_data]
            if missing:
                raise Exception(f"Missing data in state: {', '.join(missing)}")
            return combined_data
        except DeviceDisconnectedError:
            raise
        except Exception as exc:
            last_error = exc
            timed_out = _is_timeout(exc)
            if timed_out:
                timeouts += 1
            is_last = attempt >= max_retries - 1 or timeouts >= max_timeout_attempts
            delay = delays[attempt] if attempt < len(delays) else delays[-1]
            err_desc = str(exc) or type(exc).__name__
            suffix = f" (retrying in {delay:.0f}s)" if not is_last else ""
            logger.warning(
                f"get_state attempt {attempt + 1} failed: {err_desc}{suffix}"
            )
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
                except Exception as recovery_error:
                    logger.warning(f"Recovery action failed: {recovery_error}")
            if is_last:
                break
            await asyncio.sleep(delay)

    last_desc = str(last_error) or type(last_error).__name__
    error_msg = f"Failed to get state after {attempts} attempts: {last_desc}"
    logger.error(error_msg)
    raise StateFetchFailed(error_msg) from last_error
