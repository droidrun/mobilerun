import asyncio
import base64
import json
import re
import shutil
from typing import Any

import httpx
from async_adbutils import AdbDevice, adb
from mobilerun_core_local.driver.android.portal import (
    PORTAL_PACKAGE_NAME,
    check_portal_accessibility,
    get_compatible_portal_version,
    portal_content_uri,
    portal_ime_id,
)
from mobilerun_core_local.transport.android.portal_client import PortalClient

from mobilerun import __version__
from mobilerun.clients.portal.releases import get_latest_portal_version
from mobilerun.config.loader import ConfigLoader, OutdatedConfigError
from mobilerun.config.settings import get_settings
from mobilerun.diagnostics.results import CheckResult, Status
from mobilerun.diagnostics.versions import (
    get_latest_sdk_version,
    parse_portal_version,
    parse_version_tuple,
)


async def check_sdk_version(debug: bool) -> CheckResult:
    """Check Mobilerun SDK version against latest GitHub release."""
    current = __version__
    latest = get_latest_sdk_version()

    if latest is None:
        return CheckResult(
            "SDK Version",
            Status.WARN,
            f"{current} (could not check latest)",
            detail="GitHub API unreachable",
        )

    current_t = parse_version_tuple(current)
    latest_t = parse_version_tuple(latest)

    if current_t < latest_t:
        return CheckResult(
            "SDK Version",
            Status.WARN,
            f"{current} (latest: {latest})",
            detail="pip install --upgrade mobilerun",
        )
    else:
        return CheckResult("SDK Version", Status.PASS, f"{current} (up to date)")


async def check_config(debug: bool) -> tuple[CheckResult, Any]:
    """Check config file exists and loads."""
    config_path = ConfigLoader.get_user_config_path()

    if not config_path.exists():
        # Will be created on first load
        try:
            config = ConfigLoader.load()
            return (
                CheckResult(
                    "Config",
                    Status.PASS,
                    f"{config_path} (created)",
                ),
                config,
            )
        except Exception as e:
            return (
                CheckResult(
                    "Config",
                    Status.FAIL,
                    "failed to create default config",
                    detail=str(e),
                ),
                None,
            )

    try:
        config = ConfigLoader.load()
        return CheckResult("Config", Status.PASS, str(config_path)), config
    except OutdatedConfigError:
        return (
            CheckResult(
                "Config",
                Status.FAIL,
                f"{config_path} (outdated, missing _version)",
                detail="See: assets/examples/config_example.yaml in the mobilerun repo",
            ),
            None,
        )
    except Exception as e:
        return (
            CheckResult(
                "Config",
                Status.FAIL,
                f"failed to load: {e}",
                detail=str(config_path),
            ),
            None,
        )


async def check_adb(debug: bool) -> CheckResult:
    """Check ADB is available and list devices."""
    adb_path = shutil.which("adb")
    if not adb_path:
        return CheckResult(
            "ADB",
            Status.FAIL,
            "adb not found in PATH",
            detail="Install Android SDK Platform Tools",
        )

    try:
        devices = await adb.list()
        count = len(devices)
        return CheckResult(
            "ADB",
            Status.PASS,
            f"found, {count} device(s)",
            detail=adb_path,
        )
    except Exception as e:
        return CheckResult(
            "ADB",
            Status.FAIL,
            f"adb found but server error: {e}",
            detail=adb_path,
        )


async def check_device(
    serial: str | None, debug: bool
) -> tuple[CheckResult, AdbDevice | None]:
    """Check device connection."""
    try:
        device = await adb.device(serial) if serial is not None else await adb.device()
        state = await device.get_state()
        if state == "device":
            return (
                CheckResult(
                    "Device",
                    Status.PASS,
                    f"{device.serial} (online)",
                ),
                device,
            )
        else:
            return (
                CheckResult(
                    "Device",
                    Status.FAIL,
                    f"{device.serial} (state: {state})",
                ),
                None,
            )
    except Exception as e:
        msg = str(e)
        if "no devices" in msg.lower() or "not found" in msg.lower():
            return CheckResult("Device", Status.FAIL, "no device connected"), None
        return CheckResult("Device", Status.FAIL, str(e)), None


async def check_portal_installed(
    device: AdbDevice, debug: bool
) -> tuple[CheckResult, bool]:
    """Check if Portal APK is installed. Returns (result, is_installed)."""
    try:
        packages = await device.list_packages()
        if PORTAL_PACKAGE_NAME in packages:
            return CheckResult("Portal", Status.PASS, "installed"), True
        else:
            return CheckResult("Portal", Status.FAIL, "not installed"), False
    except Exception as e:
        return CheckResult("Portal", Status.FAIL, f"error: {e}"), False


async def check_portal_version(
    device: AdbDevice, debug: bool
) -> tuple[CheckResult, str | None, str | None, str | None]:
    """Check portal version and compatibility. Returns (result, installed_ver, expected_ver, download_base)."""
    # Get installed version
    installed = None
    try:
        output = await device.shell(
            f"content query --uri {portal_content_uri(PORTAL_PACKAGE_NAME, 'version')}"
        )
        if not isinstance(output, str):
            raise ValueError("ADB shell returned non-text output")
        if "result=" in output:
            json_str = output.split("result=", 1)[1].strip()
            data = json.loads(json_str)
            if data.get("status") == "success":
                installed = data.get("result") or data.get("data")
    except Exception:
        pass

    if not installed:
        return (
            CheckResult(
                "Portal Version",
                Status.WARN,
                "could not read version",
            ),
            None,
            None,
            None,
        )

    # Get latest portal version from GitHub
    latest_portal = get_latest_portal_version()

    # Get expected (compatible) version from version mapping
    expected, download_base, mapping_fetched = get_compatible_portal_version(
        __version__, debug
    )

    installed_t = parse_portal_version(installed)

    # Build message parts
    parts = [f"v{installed}"]

    if expected:
        expected_t = parse_portal_version(expected)
        if installed_t != expected_t:
            parts.append(f"expected {expected}")

    if latest_portal:
        latest_t = parse_portal_version(latest_portal)
        if installed_t < latest_t:
            parts.append(f"latest: {latest_portal}")

    msg = " → ".join(parts[:1]) + (
        " (" + ", ".join(parts[1:]) + ")" if len(parts) > 1 else ""
    )

    # Determine status
    if expected:
        expected_t = parse_portal_version(expected)
        if installed_t < expected_t:
            return (
                CheckResult("Portal Version", Status.WARN, msg),
                installed,
                expected,
                download_base,
            )
        if installed_t > expected_t:
            # Auto-setup keeps a newer Portal too.
            return (
                CheckResult(
                    "Portal Version",
                    Status.PASS,
                    f"v{installed} (newer than {expected}, pinned for mobilerun {__version__})",
                ),
                installed,
                expected,
                download_base,
            )
        # The pinned Portal is correct even when a newer release exists.
        return (
            CheckResult(
                "Portal Version",
                Status.PASS,
                f"v{installed} (pinned for mobilerun {__version__})",
            ),
            installed,
            expected,
            download_base,
        )

    if latest_portal:
        latest_t = parse_portal_version(latest_portal)
        if installed_t < latest_t:
            # Outdated but no specific compatible version required — warn
            return (
                CheckResult(
                    "Portal Version",
                    Status.WARN,
                    msg,
                    detail="pip install --upgrade mobilerun, then mobilerun device setup",
                ),
                installed,
                expected,
                download_base,
            )

    if not mapping_fetched and not latest_portal:
        return (
            CheckResult(
                "Portal Version",
                Status.WARN,
                f"v{installed} (could not check latest)",
            ),
            installed,
            None,
            None,
        )

    return (
        CheckResult(
            "Portal Version",
            Status.PASS,
            f"v{installed} (up to date)",
        ),
        installed,
        expected,
        download_base,
    )


async def check_accessibility(
    device: AdbDevice, debug: bool
) -> tuple[CheckResult, bool]:
    """Check accessibility service. Returns (result, is_enabled)."""
    try:
        enabled = await check_portal_accessibility(device, debug=debug)
        if enabled:
            return CheckResult("Accessibility", Status.PASS, "enabled"), True
        else:
            return CheckResult("Accessibility", Status.FAIL, "not enabled"), False
    except Exception as e:
        return CheckResult("Accessibility", Status.FAIL, str(e)), False


async def check_content_provider(device: AdbDevice, debug: bool) -> CheckResult:
    """Check content provider connectivity."""
    try:
        state = await device.shell(
            f"content query --uri {portal_content_uri(PORTAL_PACKAGE_NAME, 'state')}"
        )
        if not isinstance(state, str):
            raise ValueError("ADB shell returned non-text output")
        if "Row: 0 result=" in state:
            return CheckResult("Content Provider", Status.PASS, "reachable")
        else:
            return CheckResult(
                "Content Provider",
                Status.FAIL,
                "invalid response",
                detail=state[:100] if debug else "",
            )
    except Exception as e:
        return CheckResult("Content Provider", Status.FAIL, f"unreachable: {e}")


class ForwardedPortal:
    """Probe the pinged forward directly, without ADB repair or fallback."""

    def __init__(self, port: int, headers: dict[str, str]):
        self.url = f"http://localhost:{port}"
        self.headers = headers

    async def _get(self, path: str) -> httpx.Response:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.url + path, headers=self.headers, timeout=10
            )
            response.raise_for_status()
            return response

    async def get_state(self) -> dict:
        data = (await self._get("/state_full")).json()
        if data.get("status") == "error":
            raise ValueError("Portal state request failed")
        state = data.get("result", data.get("data", data))
        return json.loads(state) if isinstance(state, str) else state

    async def take_screenshot(self, *, hide_overlay: bool) -> bytes:
        response = await self._get(
            "/screenshot" if hide_overlay else "/screenshot?hideOverlay=false"
        )
        if response.headers.get("content-type", "").startswith(
            "image/"
        ) or response.content.startswith(b"\x89PNG"):
            return response.content
        data = response.json()
        if data.get("status") == "error":
            raise ValueError("Portal screenshot request failed")
        return base64.b64decode(data.get("result", data.get("data")))


class ReadyTcp(CheckResult):
    """Successful ping and the exact endpoint to use for subsequent checks."""

    def __init__(self, port: int, headers: dict[str, str], detail: str):
        super().__init__("TCP Mode", Status.PASS, f"localhost:{port}", detail)
        self.portal = ForwardedPortal(port, headers)


async def check_tcp(
    device: AdbDevice, debug: bool, *, fix: bool = False
) -> CheckResult:
    """Check existing TCP wiring; configure server/forwarding only with fix."""

    steps = []

    # Step 1: Enable socket server via content provider
    if fix:
        try:
            toggle_uri = portal_content_uri(PORTAL_PACKAGE_NAME, "toggle_socket_server")
            await device.shell(
                f"content insert --uri {toggle_uri} --bind enabled:b:true"
            )
            steps.append("server enabled")
        except Exception as e:
            return CheckResult(
                "TCP Mode", Status.FAIL, "failed to enable socket server", detail=str(e)
            )
        await asyncio.sleep(1)

    # Step 2: Fetch auth token via content provider
    auth_headers: dict[str, str] = {}
    try:
        portal = PortalClient(device, prefer_tcp=False)
        token = await portal._fetch_auth_token()
        if token:
            auth_headers["Authorization"] = f"Bearer {token}"
            steps.append("auth token ok")
        else:
            steps.append("no auth token")
    except Exception as e:
        steps.append(f"auth token failed: {e}")

    # Step 3: Set up port forward
    local_port = None
    try:
        # Check for existing forward first
        forwards = []
        async for fwd in device.forward_list():
            forwards.append(fwd)
        for fwd in forwards:
            if (
                fwd.serial == device.serial
                and fwd.remote == f"tcp:{get_settings().MOBILERUN_PORTAL_REMOTE_PORT}"
            ):
                match = re.search(r"tcp:(\d+)", fwd.local)
                if match:
                    local_port = int(match.group(1))
                    steps.append(f"reusing forward :{local_port}")
                    break

        if local_port is None:
            if not fix:
                return CheckResult(
                    "TCP Mode",
                    Status.WARN,
                    "no existing port forward; use doctor --fix to configure TCP",
                )
            local_port = await device.forward_port(
                get_settings().MOBILERUN_PORTAL_REMOTE_PORT
            )
            steps.append(f"forwarded :{local_port}")
    except Exception as e:
        return CheckResult(
            "TCP Mode",
            Status.FAIL,
            "port forward failed",
            detail=str(e),
        )

    # Step 4: Test connection with auth
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"http://localhost:{local_port}/ping",
                headers=auth_headers,
                timeout=5,
            )
            if resp.status_code == 200:
                steps.append("ping ok")
                return ReadyTcp(local_port, auth_headers, ", ".join(steps))
            else:
                return CheckResult(
                    "TCP Mode",
                    Status.FAIL,
                    f"ping returned {resp.status_code}",
                    detail=", ".join(steps),
                )
    except Exception as e:
        return CheckResult(
            "TCP Mode",
            Status.FAIL,
            f"ping failed: {e}",
            detail=", ".join(steps),
        )


async def check_portal_state(
    device: AdbDevice, debug: bool, *, tcp_portal: ForwardedPortal | None = None
) -> tuple[CheckResult, Any]:
    """Fetch full state and verify a11y_tree, phone_state, device_context are present."""
    try:
        portal: PortalClient | ForwardedPortal
        if tcp_portal is None:
            portal = PortalClient(device, prefer_tcp=False)
            await portal.connect()
            method = "content_provider"
        else:
            portal = tcp_portal
            method = "tcp"

        state = await portal.get_state()

        if isinstance(state, dict) and state.get("error"):
            return (
                CheckResult(
                    "Portal State",
                    Status.FAIL,
                    f"error: {state.get('message', state.get('error'))}",
                ),
                portal,
            )

        required = ("a11y_tree", "phone_state", "device_context")
        present = [k for k in required if k in state]
        missing = [k for k in required if k not in state]

        if missing:
            return (
                CheckResult(
                    "Portal State",
                    Status.FAIL,
                    f"missing: {', '.join(missing)}",
                    detail=f"present: {', '.join(present)} (via {method})",
                ),
                portal,
            )
        else:
            return (
                CheckResult(
                    "Portal State",
                    Status.PASS,
                    f"a11y_tree, phone_state, device_context (via {method})",
                ),
                portal,
            )
    except Exception as e:
        return CheckResult("Portal State", Status.FAIL, str(e)), None


async def check_screenshot(portal: Any, debug: bool) -> CheckResult:
    """Take a test screenshot to verify the full capture pipeline."""
    try:
        data = await portal.take_screenshot(hide_overlay=True)
        if data and len(data) > 0:
            size_kb = len(data) / 1024
            return CheckResult(
                "Screenshot",
                Status.PASS,
                f"ok ({size_kb:.0f} KB)",
            )
        else:
            return CheckResult(
                "Screenshot",
                Status.FAIL,
                "empty response",
            )
    except Exception as e:
        return CheckResult(
            "Screenshot",
            Status.FAIL,
            f"capture failed: {e}",
        )


async def check_keyboard(device: AdbDevice, debug: bool) -> CheckResult:
    """Check if Mobilerun keyboard IME is available."""
    try:
        output = await device.shell("ime list -s")
        if not isinstance(output, str):
            raise ValueError("ADB shell returned non-text output")
        ime_id = portal_ime_id(PORTAL_PACKAGE_NAME)
        if ime_id in output:
            return CheckResult("Keyboard", Status.PASS, "available")
        else:
            return CheckResult(
                "Keyboard",
                Status.WARN,
                "not listed",
                detail="Will be set up automatically when needed",
            )
    except Exception as e:
        return CheckResult("Keyboard", Status.WARN, f"could not check: {e}")
