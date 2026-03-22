"""DeviceDriver — raw device I/O interface.

Subclasses implement the actual communication (ADB, iOS HTTP, cloud SDK, etc.).
Unsupported methods are detected via the ``supported`` set, not introspection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DeviceDisconnectedError(Exception):
    """Raised when the device is no longer reachable."""

    pass


class DeviceDriver:
    """Base class for all device drivers.

    Every method raises ``NotImplementedError`` by default.
    Concrete drivers override the methods they support and declare them
    in the ``supported`` class-level set.
    """

    supported: set[str] = set()
    supported_system_buttons: set[str] = set()
    device_prompt_name: str = "device"

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        """Establish connection to the device."""
        raise NotImplementedError

    async def ensure_connected(self) -> None:
        """Connect if not already connected."""
        raise NotImplementedError

    # -- input actions -------------------------------------------------------

    async def tap(self, x: int, y: int) -> None:
        """Tap at absolute pixel coordinates."""
        raise NotImplementedError

    async def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: float = 1000,
    ) -> None:
        """Swipe from (x1, y1) to (x2, y2)."""
        raise NotImplementedError

    async def input_text(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0,
    ) -> bool:
        """Type *text* into the currently focused field.

        Returns ``True`` on success, ``False`` on failure.
        """
        raise NotImplementedError

    async def press_key(self, keycode: int) -> None:
        """Send a single key-event."""
        raise NotImplementedError

    async def press_system_button(self, button: str) -> None:
        """Press a semantic system button like ``back`` or ``home``."""
        raise NotImplementedError

    async def drag(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: float = 3.0,
    ) -> None:
        """Drag from (x1, y1) to (x2, y2)."""
        raise NotImplementedError

    # -- app management ------------------------------------------------------

    async def start_app(self, package: str, activity: Optional[str] = None) -> str:
        """Launch an application.

        Returns a human-readable result string.
        """
        raise NotImplementedError

    async def install_app(self, path: str, **kwargs) -> str:
        """Install an APK/IPA at *path*."""
        raise NotImplementedError

    async def get_apps(self, include_system: bool = True) -> List[Dict[str, str]]:
        """Return installed apps as ``[{"package": …, "label": …}, …]``."""
        raise NotImplementedError

    async def list_packages(self, include_system: bool = False) -> List[str]:
        """Return installed package names."""
        raise NotImplementedError

    # -- state / observation -------------------------------------------------

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        """Capture the current screen.

        Returns raw PNG bytes.
        """
        raise NotImplementedError

    async def get_ui_tree(self) -> Dict[str, Any]:
        """Return the raw UI / accessibility tree from the device."""
        raise NotImplementedError

    async def get_date(self) -> str:
        """Return the device's current date/time as a string."""
        raise NotImplementedError
