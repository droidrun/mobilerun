"""Portal auto-setup helpers."""

import inspect
from typing import Any, Callable

from mobilerun import __version__


def portal_version_kwargs(
    setup_fn: Callable[..., Any], *, in_place: bool = False
) -> dict[str, Any]:
    """Key the Portal version map by mobilerun's version when core-local allows it.

    With in_place, also ask for an update that keeps Portal's data.
    """
    params = inspect.signature(setup_fn).parameters
    kwargs: dict[str, Any] = {}
    if "version" in params:
        kwargs["version"] = __version__
    if in_place and "uninstall" in params:
        kwargs["uninstall"] = False
    return kwargs


async def adb_install_verifier_enabled(device: Any) -> bool:
    """Whether the device verifies apps installed over adb ("Verify apps over USB").

    When on, Google Play Protect can block the Portal install until the user
    taps "Install anyway" on the device. Unknown or unreadable counts as off.
    """
    try:
        value = await device.shell("settings get global verifier_verify_adb_installs")
    except Exception:
        return False
    return str(value).strip() == "1"
