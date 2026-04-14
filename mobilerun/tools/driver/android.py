"""AndroidDriver — ADB-based device driver.

Wraps ``adbutils.Device`` + ``PortalClient`` to provide clean device I/O
without event emission, formatting, or element lookup.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from async_adbutils import adb

from droidrun.tools.android.portal_client import PortalClient
from droidrun.tools.driver.base import DeviceDriver

logger = logging.getLogger("droidrun")

PORTAL_DEFAULT_TCP_PORT = 8080


class AndroidDriver(DeviceDriver):
    """Raw Android device I/O via ADB + Portal."""

    platform = "Android"

    supported = {
        "tap",
        "swipe",
        "input_text",
        "press_button",
        "start_app",
        "screenshot",
        "get_ui_tree",
        "get_date",
        "get_apps",
        "list_packages",
        "install_app",
        "drag",
    }

    supported_buttons = {"back", "home", "enter"}

    _BUTTON_KEYCODES = {
        "back": 4,
        "home": 3,
        "enter": 66,
    }

    def __init__(
        self,
        serial: str | None = None,
        use_tcp: bool = False,
        remote_tcp_port: int = PORTAL_DEFAULT_TCP_PORT,
    ) -> None:
        self._serial = serial
        self._use_tcp = use_tcp
        self._remote_tcp_port = remote_tcp_port
        self.device = None
        self.portal: PortalClient | None = None
        self._connected = False

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        if self._connected:
            return

        self.device = await adb.device(serial=self._serial)
        state = await self.device.get_state()
        if state != "device":
            raise ConnectionError(f"Device is not online. State: {state}")

        self.portal = PortalClient(self.device, prefer_tcp=self._use_tcp)
        await self.portal.connect()

        from droidrun.portal import setup_keyboard  # circular import guard

        await setup_keyboard(self.device)
        self._connected = True

    async def ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    # -- input actions -------------------------------------------------------

    async def tap(self, x: int, y: int) -> None:
        await self.ensure_connected()
        await self.device.click(x, y)

    async def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: float = 1000,
    ) -> None:
        await self.ensure_connected()
        await self.device.swipe(x1, y1, x2, y2, float(duration_ms / 1000))
        await asyncio.sleep(duration_ms / 1000)

    async def input_text(self, text: str, clear: bool = False) -> bool:
        await self.ensure_connected()
        return await self.portal.input_text(text, clear)

    async def press_button(self, button: str) -> None:
        await self.ensure_connected()
        button_lower = button.lower()
        if button_lower not in self.supported_buttons:
            raise ValueError(
                f"Button '{button}' not supported. "
                f"Supported: {', '.join(sorted(self.supported_buttons))}"
            )
        await self.device.keyevent(self._BUTTON_KEYCODES[button_lower])

    async def drag(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: float = 3.0,
    ) -> None:
        await self.ensure_connected()
        raise NotImplementedError("Drag is not implemented yet")

    # -- app management ------------------------------------------------------

    async def start_app(self, package: str, activity: Optional[str] = None) -> str:
        await self.ensure_connected()
        try:
            logger.debug(f"Starting app {package} with activity {activity}")
            if not activity:
                dumpsys_output = await self.device.shell(
                    f"cmd package resolve-activity --brief {package}"
                )
                activity = dumpsys_output.splitlines()[1].split("/")[1]

            logger.debug(f"Activity: {activity}")
            await self.device.app_start(package, activity)
            logger.debug(f"App started: {package} with activity {activity}")
            return f"App started: {package} with activity {activity}"
        except Exception as e:
            return f"Failed to start app {package}: {e}"

    async def install_app(self, path: str, **kwargs) -> str:
        await self.ensure_connected()
        if not os.path.exists(path):
            return f"Failed to install app: APK file not found at {path}"

        reinstall = kwargs.get("reinstall", False)
        grant_permissions = kwargs.get("grant_permissions", True)

        logger.debug(
            f"Installing app: {path} with reinstall: {reinstall} "
            f"and grant_permissions: {grant_permissions}"
        )
        result = await self.device.install(
            path,
            nolaunch=True,
            uninstall=reinstall,
            flags=["-g"] if grant_permissions else [],
            silent=True,
        )
        logger.debug(f"Installed app: {path} with result: {result}")
        return result

    async def get_apps(self, include_system: bool = True) -> List[Dict[str, str]]:
        await self.ensure_connected()
        return await self.portal.get_apps(include_system)

    async def list_packages(self, include_system: bool = False) -> List[str]:
        await self.ensure_connected()
        filter_list = [] if include_system else ["-3"]
        return await self.device.list_packages(filter_list)

    # -- state / observation -------------------------------------------------

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        await self.ensure_connected()
        return await self.portal.take_screenshot(hide_overlay)

    async def get_ui_tree(self) -> Dict[str, Any]:
        await self.ensure_connected()
        return await self.portal.get_state()

    async def get_date(self) -> str:
        await self.ensure_connected()
        result = await self.device.shell("date")
        return result.strip()
