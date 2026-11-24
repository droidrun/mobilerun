"""Android user targeting for Portal commands the CLI builds itself.

Each Android user runs its own Portal. ``content`` and ``pm list packages``
default to user 0, so with a secondary user in front they reach a Portal whose
accessibility service is unbound. ``settings``, ``ime`` and ``am start``
already follow the foreground user and need no flag.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("mobilerun")


async def _read_foreground_user(device: Any) -> int | None:
    """Return the foreground Android user id, or ``None`` if unreadable."""
    try:
        output = await device.shell("am get-current-user")
    except Exception as e:
        logger.debug(f"Could not read the foreground Android user: {e}")
        return None
    match = re.fullmatch(r"\s*(\d+)\s*", output) if isinstance(output, str) else None
    return int(match.group(1)) if match else None


def _user_flag(user: int | None) -> str:
    """Return `` --user N`` for a secondary user, else an empty string.

    User 0 and an unknown user get no flag, so commands on single-user devices
    stay exactly as before.
    """
    if isinstance(user, int) and not isinstance(user, bool) and user > 0:
        return f" --user {user}"
    return ""


try:
    from mobilerun_core_local.driver.android.portal import (
        get_foreground_user,
        user_flag,
    )
except ImportError:  # mobilerun-core-local without multi-user support
    get_foreground_user = _read_foreground_user
    user_flag = _user_flag


async def foreground_user_flag(device: Any) -> str:
    """`` --user N`` for the foreground user, or ``""`` for user 0."""
    return user_flag(await get_foreground_user(device))


async def list_foreground_packages(device: Any) -> list[str]:
    """``pm list packages`` for the foreground user."""
    user = await get_foreground_user(device)
    if user_flag(user):
        return await device.list_packages(["--user", str(user)])
    return await device.list_packages()


__all__ = [
    "foreground_user_flag",
    "get_foreground_user",
    "list_foreground_packages",
    "user_flag",
]
