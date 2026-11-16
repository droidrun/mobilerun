"""
Anonymous telemetry tracking using PostHog.

This module handles opt-in telemetry collection to help improve Mobilerun.
All data is anonymized, and telemetry can be disabled through the Mobilerun
configuration or environment variables.
"""

import asyncio
import logging
import os
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4

from posthog import Posthog

from mobilerun.telemetry.events import TelemetryEvent

logger = logging.getLogger("mobilerun-telemetry")
mobilerun_logger = logging.getLogger("mobilerun")

PROJECT_API_KEY = "phc_XyD3HKIsetZeRkmnfaBughs8fXWYArSUFc30C0HmRiO"
HOST = "https://eu.i.posthog.com"
USER_ID_PATH = Path.home() / ".droidrun" / "user_id"
RUN_ID = str(uuid4())

TELEMETRY_ENABLED_MESSAGE = "Anonymized telemetry enabled. See https://docs.mobilerun.ai/v3/guides/telemetry for more information."
TELEMETRY_DISABLED_MESSAGE = "🛑 Anonymized telemetry disabled. Telemetry can be controlled through Mobilerun configuration or the MOBILERUN_TELEMETRY_ENABLED environment variable."

# Created lazily on first capture/flush. Constructing the PostHog client spawns
# a consumer thread and registers a blocking atexit flush, which added ~5s to
# the shutdown of EVERY command (incl. `mobilerun --help`) because importing
# mobilerun imports this module. Deferring it keeps non-telemetry commands fast.
_posthog: Posthog | None = None
_posthog_lock = Lock()


def _get_posthog() -> Posthog:
    global _posthog
    if _posthog is None:
        with _posthog_lock:
            if _posthog is None:
                _posthog = Posthog(
                    project_api_key=PROJECT_API_KEY,
                    host=HOST,
                    disable_geoip=False,
                )
    return _posthog


def is_telemetry_enabled(*, config_enabled: bool = True) -> bool:
    """
    Check whether both configuration and environment policy enable telemetry.

    Returns:
        True when ``config_enabled`` is true and the telemetry
        environment variable is true/1/yes/y (case-insensitive). Environment
        policy defaults to enabled when the variable is unset.
    """
    telemetry_enabled = os.environ.get("MOBILERUN_TELEMETRY_ENABLED")
    if telemetry_enabled is None:
        telemetry_enabled = "true"
    enabled = config_enabled and telemetry_enabled.lower() in ["true", "1", "yes", "y"]
    logger.debug(f"Telemetry enabled: {enabled}")
    return enabled


def print_telemetry_message(*, config_enabled: bool = True) -> None:
    """
    Print telemetry status message to the logger.

    Displays the effective status based on configuration and environment policy.
    """
    if is_telemetry_enabled(config_enabled=config_enabled):
        mobilerun_logger.debug(TELEMETRY_ENABLED_MESSAGE)

    else:
        mobilerun_logger.debug(TELEMETRY_DISABLED_MESSAGE)


# Print telemetry message on import
print_telemetry_message()


def _is_valid_uuid(value: str) -> bool:
    """Check if string is a valid UUID format."""
    try:
        UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def get_user_id() -> str:
    """
    Get or create persistent anonymous user ID.

    The user ID is stored in ~/.droidrun/user_id and persists across sessions.
    If the file doesn't exist or contains an invalid UUID, a new one is generated.

    Returns:
        User UUID string, or "unknown" if an error occurs.
    """
    try:
        # Ensure directory exists
        USER_ID_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Read existing ID if valid
        if USER_ID_PATH.exists():
            user_id = USER_ID_PATH.read_text().strip()

            # Validate UUID format
            if user_id and _is_valid_uuid(user_id):
                logger.debug(f"User ID: {user_id}")
                return user_id
            else:
                logger.debug(f"Invalid user ID found in {USER_ID_PATH}, regenerating")

        # Generate new UUID (file missing or invalid)
        user_id = str(uuid4())
        USER_ID_PATH.write_text(user_id)
        logger.debug(f"Generated new user ID: {user_id}")
        return user_id

    except Exception as e:
        logger.error(f"Error getting user ID: {e}")
        return "unknown"


def capture(
    event: TelemetryEvent,
    user_id: str | None = None,
    *,
    config_enabled: bool = True,
) -> None:
    """
    Capture and send a telemetry event to PostHog.

    Args:
        event: Telemetry event to capture (must be a TelemetryEvent subclass)
        user_id: Optional user ID to use instead of the default persistent ID
        config_enabled: Whether telemetry is enabled in Mobilerun configuration

    Note:
        This function is a no-op if telemetry is disabled.
    """
    if not is_telemetry_enabled(config_enabled=config_enabled):
        logger.debug(f"Telemetry disabled, skipping capture of {event}")
        return

    try:
        event_name = type(event).__name__
        event_data = event.model_dump()
        properties = {
            "run_id": RUN_ID,
            **event_data,
        }

        _get_posthog().capture(
            event_name, distinct_id=user_id or get_user_id(), properties=properties
        )
        logger.debug(f"Captured event: {event_name} with properties: {event}")
    except Exception as e:
        logger.error(f"Error capturing event: {e}")


async def flush(*, config_enabled: bool = True) -> None:
    try:
        if not is_telemetry_enabled(config_enabled=config_enabled) or _posthog is None:
            return

        await asyncio.wait_for(asyncio.to_thread(_posthog.flush), timeout=10)
    except asyncio.TimeoutError:
        logger.warning("PostHog flush timed out after 10 seconds")
    except Exception as e:
        logger.error(f"Error flushing data: {e}")
