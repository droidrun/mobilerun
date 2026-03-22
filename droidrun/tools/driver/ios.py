"""IOSDriver — HTTP REST-based device driver for iOS.

Wraps the iOS portal HTTP API (running on the device) to provide device I/O
through the same ``DeviceDriver`` interface used by Android.

Known limitations:
- ``get_apps`` returns bundle identifiers, not real app metadata
- ``drag`` not implemented (same as Android)
- ``install_app`` not supported
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from droidrun.tools.driver.base import DeviceDriver

logger = logging.getLogger("droidrun")

SYSTEM_BUNDLE_IDENTIFIERS = [
    "ai.droidrun.droidrun-ios-portal",
    "com.apple.Bridge",
    "com.apple.DocumentsApp",
    "com.apple.Fitness",
    "com.apple.Health",
    "com.apple.Maps",
    "com.apple.MobileAddressBook",
    "com.apple.MobileSMS",
    "com.apple.Passbook",
    "com.apple.Passwords",
    "com.apple.Preferences",
    "com.apple.PreviewShell",
    "com.apple.mobilecal",
    "com.apple.mobilesafari",
    "com.apple.mobileslideshow",
    "com.apple.news",
    "com.apple.reminders",
    "com.apple.shortcuts",
    "com.apple.webapp",
]

# Android keycode → iOS action mapping.
_ANDROID_TO_IOS_KEYCODE = {
    3: 0,  # HOME → iOS key 0
}


class IOSDriver(DeviceDriver):
    """iOS device driver communicating via HTTP REST to the iOS portal app."""

    supported = {
        "tap",
        "swipe",
        "input_text",
        "press_key",
        "start_app",
        "screenshot",
        "get_ui_tree",
        "get_date",
        "list_packages",
        "get_apps",
    }
    supported_system_buttons = {"back", "home", "enter"}
    device_prompt_name = "iPhone"

    def __init__(
        self,
        url: str,
        bundle_identifiers: Optional[List[str]] = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.bundle_identifiers = bundle_identifiers or []
        self._client: Optional[httpx.AsyncClient] = None
        self._last_tapped_rect: Optional[str] = None
        self._connected = False

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(base_url=self.url, timeout=30.0)
        self._connected = True
        logger.info(f"Connected to iOS device at {self.url}")

    async def ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    # -- input actions -------------------------------------------------------

    async def tap(self, x: int, y: int) -> None:
        ios_rect = f"{{{{{x},{y}}},{{{1},{1}}}}}"
        self._last_tapped_rect = f"{x},{y},1,1"
        resp = await self._client.post(
            "/gestures/tap",
            json={"rect": ios_rect, "count": 1, "longPress": False},
        )
        resp.raise_for_status()

    async def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        resp = await self._client.post(
            "/gestures/swipe",
            json={
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "durationMs": float(duration_ms),
            },
        )
        resp.raise_for_status()

    async def input_text(self, text: str, clear: bool = False) -> bool:
        if clear and self._last_tapped_rect:
            x, y, w, h = self._last_tapped_rect.split(",")
            ios_rect = f"{{{{{x},{y}}},{{{w},{h}}}}}"
            await self._client.post("/inputs/clear", json={"rect": ios_rect})
        resp = await self._client.post(
            "/inputs/type_focused", json={"text": text}
        )
        return resp.status_code == 200

    async def press_key(self, keycode: int) -> None:
        if keycode == 4:  # BACK
            resp = await self._client.post("/gestures/back")
            resp.raise_for_status()
            return
        if keycode == 66:  # ENTER
            resp = await self._client.post(
                "/inputs/type_focused", json={"text": "\n"}
            )
            resp.raise_for_status()
            return
        # HOME and other mapped keys
        ios_keycode = _ANDROID_TO_IOS_KEYCODE.get(keycode)
        if ios_keycode is None:
            logger.warning(f"Keycode {keycode} not supported on iOS, ignoring")
            return
        resp = await self._client.post("/inputs/key", json={"key": ios_keycode})
        resp.raise_for_status()

    async def press_system_button(self, button: str) -> None:
        keycodes = {"back": 4, "home": 3, "enter": 66}
        keycode = keycodes.get(button.lower())
        if keycode is None:
            raise ValueError(f"Unsupported system button: {button}")
        await self.press_key(keycode)

    # -- app management ------------------------------------------------------

    async def start_app(self, package: str, activity: Optional[str] = None) -> str:
        resp = await self._client.post(
            "/inputs/launch", json={"bundleIdentifier": package}
        )
        if resp.status_code == 200:
            return f"Launched {package}"
        return f"Failed to launch {package}: HTTP {resp.status_code}"

    async def get_apps(self, include_system: bool = True) -> List[Dict[str, str]]:
        all_ids: set[str] = set(self.bundle_identifiers)
        if include_system:
            all_ids.update(SYSTEM_BUNDLE_IDENTIFIERS)
        return [{"package": bid, "label": bid} for bid in sorted(all_ids)]

    async def list_packages(self, include_system: bool = False) -> List[str]:
        apps = await self.get_apps(include_system)
        return [a["package"] for a in apps]

    # -- state / observation -------------------------------------------------

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        resp = await self._client.get("/vision/screenshot")
        resp.raise_for_status()
        return resp.content

    async def get_ui_tree(self) -> Dict[str, Any]:
        """Return raw iOS accessibility data, phone state, and screen size."""
        a11y_resp = await self._client.get("/vision/a11y")
        a11y_resp.raise_for_status()
        a11y_data = a11y_resp.json()

        state_resp = await self._client.get("/vision/state")
        if state_resp.status_code == 200:
            state_data = state_resp.json()
            phone_state = {
                "currentApp": state_data.get("activity", "Unknown"),
                "keyboardVisible": state_data.get("keyboardShown", False),
                "focusedElement": state_data.get("focusedElement"),
            }
        else:
            phone_state = {
                "currentApp": "Unknown",
                "keyboardVisible": False,
                "focusedElement": None,
            }

        screen_resp = await self._client.get("/device/screen")
        screen_data = (
            screen_resp.json() if screen_resp.status_code == 200 else {}
        )

        return {
            "a11y_raw": a11y_data["accessibilityTree"],
            "phone_state": phone_state,
            "screen_size": screen_data,
        }

    async def get_date(self) -> str:
        await self.ensure_connected()
        resp = await self._client.get("/device/date")
        if resp.status_code == 200:
            return resp.json().get("date", "")
        return ""
