"""IOSDriver — HTTP REST-based device driver for iOS.

Wraps the iOS portal HTTP API (running on the device) to provide device I/O
through the same ``DeviceDriver`` interface used by Android.

Known limitations:
- ``get_apps`` returns a hardcoded list of system bundle identifiers
- ``packageName`` is not tracked (iOS has no API to detect the foreground app)
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from droidrun.tools.driver.base import DeviceDriver

logger = logging.getLogger("droidrun")

SYSTEM_APP_LABELS = {
    "ai.droidrun.droidrun-ios-portal": "Droidrun Portal",
    "com.apple.Bridge": "Watch",
    "com.apple.DocumentsApp": "Files",
    "com.apple.Fitness": "Fitness",
    "com.apple.Health": "Health",
    "com.apple.Maps": "Maps",
    "com.apple.MobileAddressBook": "Contacts",
    "com.apple.MobileSMS": "Messages",
    "com.apple.Passbook": "Wallet",
    "com.apple.Passwords": "Passwords",
    "com.apple.Preferences": "Settings",
    "com.apple.PreviewShell": "Freeform",
    "com.apple.mobilecal": "Calendar",
    "com.apple.mobilesafari": "Safari",
    "com.apple.mobileslideshow": "Photos",
    "com.apple.news": "News",
    "com.apple.reminders": "Reminders",
    "com.apple.shortcuts": "Shortcuts",
    "com.apple.webapp": "Web App",
}


IOS_PORTAL_DEFAULT_PORT = 6643
IOS_PORTAL_SCAN_RANGE = 10
IOS_STATE_TIMEOUT_SECONDS = 4.0
IOS_STATE_HTTP_TIMEOUT_SECONDS = 6.0


def validate_ios_portal_url(url: str) -> str:
    """Validate and normalize an iOS portal base URL."""
    normalized = url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "iOS device must be the portal base URL, e.g. http://127.0.0.1:6643"
        )
    return normalized


async def discover_ios_portal(
    host: str = "127.0.0.1",
    start_port: int = IOS_PORTAL_DEFAULT_PORT,
    scan_range: int = IOS_PORTAL_SCAN_RANGE,
    timeout: float = 1.0,
) -> str:
    """Auto-discover the iOS portal by scanning a small port range.

    Tries ``start_port`` first (fast path), then scans the remaining ports
    in ``[start_port, start_port + scan_range)`` concurrently.

    Returns:
        The portal base URL, e.g. ``http://127.0.0.1:6643``.

    Raises:
        ConnectionError: If no portal is found in the scan range.
    """

    async def _probe(client: httpx.AsyncClient, port: int) -> Optional[str]:
        url = f"http://{host}:{port}"
        try:
            resp = await client.get(f"{url}/device/date")
            if resp.status_code == 200 and "date" in resp.json():
                return url
        except Exception:
            pass
        return None

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Fast path: try the default port first
        result = await _probe(client, start_port)
        if result:
            logger.info(f"iOS portal found at {result}")
            return result

        # Scan remaining ports concurrently
        results = await asyncio.gather(
            *[_probe(client, p) for p in range(start_port + 1, start_port + scan_range)]
        )
        for r in results:
            if r is not None:
                logger.info(f"iOS portal found at {r}")
                return r

    raise ConnectionError(
        f"Could not find iOS portal on {host} "
        f"(scanned ports {start_port}-{start_port + scan_range - 1}). "
        "Make sure the Droidrun Portal app is running and iproxy is forwarding the port."
    )


def _humanize_bundle_identifier(bundle_id: str) -> str:
    mapped = SYSTEM_APP_LABELS.get(bundle_id)
    if mapped:
        return mapped

    last_segment = bundle_id.rsplit(".", 1)[-1]
    words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+", last_segment)
    if words:
        return " ".join(words)
    return last_segment or bundle_id


class IOSDriver(DeviceDriver):
    """iOS device driver communicating via HTTP REST to the iOS portal app."""

    platform = "iOS"

    supported = {
        "tap",
        "swipe",
        "input_text",
        "press_button",
        "start_app",
        "screenshot",
        "get_ui_tree",
        "list_packages",
        "get_apps",
        "get_date",
    }

    supported_buttons = {"home"}

    def __init__(
        self,
        url: str,
        bundle_identifiers: Optional[List[str]] = None,
    ) -> None:
        self.url = validate_ios_portal_url(url)
        self.bundle_identifiers = bundle_identifiers or []
        self._client: Optional[httpx.AsyncClient] = None
        self._connected = False

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(base_url=self.url, timeout=30.0)
        try:
            resp = await self._client.get("/device/date")
            resp.raise_for_status()
        except Exception as exc:
            await self._client.aclose()
            self._client = None
            raise ConnectionError(
                f"Could not connect to iOS portal at {self.url}. "
                "Make sure the Droidrun Portal app is running on the device "
                "and the URL/port is correct."
            ) from exc
        self._connected = True
        logger.info(f"Connected to iOS device at {self.url}")

    async def ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    # -- input actions -------------------------------------------------------

    async def tap(self, x: int, y: int) -> None:
        ios_rect = f"{{{{{x},{y}}},{{{1},{1}}}}}"
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
        resp = await self._client.post(
            "/inputs/type", json={"text": text, "clear": clear}
        )
        return resp.status_code == 200

    async def press_button(self, button: str) -> None:
        await self.ensure_connected()
        button_lower = button.lower()
        if button_lower not in self.supported_buttons:
            raise ValueError(
                f"Button '{button}' not supported on iOS. "
                f"Supported: {', '.join(sorted(self.supported_buttons))}"
            )
        if button_lower == "home":
            resp = await self._client.post("/inputs/key", json={"key": 1})
            resp.raise_for_status()
            return

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
            all_ids.update(SYSTEM_APP_LABELS)
        return [
            {"package": bid, "label": _humanize_bundle_identifier(bid)}
            for bid in sorted(all_ids)
        ]

    async def list_packages(self, include_system: bool = False) -> List[str]:
        apps = await self.get_apps(include_system)
        return [a["package"] for a in apps]

    # -- state / observation -------------------------------------------------

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        resp = await self._client.get("/vision/screenshot")
        resp.raise_for_status()
        return resp.content

    async def get_ui_tree(self) -> Dict[str, Any]:
        """Return unified state from the iOS portal.

        Returns a dict with ``a11y_tree``, ``phone_state``, and
        ``device_context`` keys — matching the format expected by
        ``fetch_state_with_retry()``.
        """
        resp = await self._client.get(
            "/state",
            params={"timeout": str(IOS_STATE_TIMEOUT_SECONDS)},
            timeout=IOS_STATE_HTTP_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception as e:
            raise ValueError(f"Invalid response from /state: {e}") from e

    async def get_date(self) -> str:
        resp = await self._client.get("/device/date")
        if resp.status_code == 200:
            return resp.json().get("date", "")
        return ""
