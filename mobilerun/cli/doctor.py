"""Mobilerun Doctor - System health checks and diagnostics."""

import asyncio
import json
import re
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Any

import requests
from async_adbutils import AdbDevice, adb
from rich.console import Console
import httpx
from mobilerun.tools.android.portal_client import PortalClient

from mobilerun import __version__
from mobilerun.portal import (
    LEGACY_PORTAL_PACKAGE_NAME,
    PORTAL_PACKAGE_NAME,
    GITHUB_API_HOSTS,
    REPO,
    check_portal_accessibility,
    enable_portal_accessibility,
    get_compatible_portal_version,
    portal_content_uri,
    portal_ime_id,
)

console = Console()

PORTAL_REMOTE_PORT = 8080


class Status(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str
    detail: str = ""


def _status_icon(status: Status) -> str:
    if status == Status.PASS:
        return "[green]✓[/]"
    elif status == Status.WARN:
        return "[yellow]⚠[/]"
    else:
        return "[red]✗[/]"


def _print_result(result: CheckResult, debug: bool = False) -> None:
    icon = _status_icon(result.status)
    console.print(f"  {result.name:<22} {result.message:<42} {icon}")
    if debug and result.detail:
        console.print(f"  {'':22} [dim]{result.detail}[/]")


def _parse_version_tuple(version_str: str) -> tuple:
    """Parse a version string like '0.4.23' or 'v0.4.23' into a comparable tuple."""
    cleaned = version_str.lstrip("v").strip()
    parts = []
    for p in cleaned.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts)


# ── Check Functions ──────────────────────────────────────────────


async def check_sdk_version(debug: bool) -> CheckResult:
    """Check Mobilerun SDK version against latest GitHub release."""
    current = __version__
    latest = None

    for host in GITHUB_API_HOSTS:
        try:
            url = f"{host}/repos/mobilerun/mobilerun/releases/latest"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # Handle both GitHub API formats
                if "release" in data:
                    tag = data["release"].get(
                        "tag", data["release"].get("tag_name", "")
                    )
                else:
                    tag = data.get("tag_name", "")
                latest = tag.lstrip("v")
                break
        except Exception:
            continue

    if latest is None:
        return CheckResult(
            "SDK Version",
            Status.WARN,
            f"{current} (could not check latest)",
            detail="GitHub API unreachable",
        )

    current_t = _parse_version_tuple(current)
    latest_t = _parse_version_tuple(latest)

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
    from mobilerun.config_manager import ConfigLoader
    from mobilerun.config_manager.loader import OutdatedConfigError

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
                detail="See: mobilerun config_example.yaml",
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
        device = await adb.device(serial)
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


def _get_latest_portal_version() -> str | None:
    """Fetch latest portal release version from GitHub."""
    for host in GITHUB_API_HOSTS:
        try:
            url = f"{host}/repos/{REPO}/releases/latest"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if "release" in data:
                    tag = data["release"].get(
                        "tag", data["release"].get("tag_name", "")
                    )
                else:
                    tag = data.get("tag_name", "")
                return tag.lstrip("v").strip() if tag else None
        except Exception:
            continue
    return None


async def check_portal_version(
    device: AdbDevice, debug: bool
) -> tuple[CheckResult, str | None, str | None, str | None]:
    """Check portal version and compatibility. Returns (result, installed_ver, expected_ver, download_base)."""
    # Get installed version
    installed = None
    try:
        output = await device.shell(
            f"content query --uri {portal_content_uri(LEGACY_PORTAL_PACKAGE_NAME, 'version')}"
        )
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
    latest_portal = _get_latest_portal_version()

    # Get expected (compatible) version from version mapping
    expected, download_base, mapping_fetched = get_compatible_portal_version(
        __version__, debug
    )

    installed_t = _parse_version_tuple(installed)

    # Build message parts
    parts = [f"v{installed}"]

    if expected:
        expected_t = _parse_version_tuple(expected)
        if installed_t != expected_t:
            parts.append(f"expected {expected}")

    if latest_portal:
        latest_t = _parse_version_tuple(latest_portal)
        if installed_t < latest_t:
            parts.append(f"latest: {latest_portal}")

    msg = " → ".join(parts[:1]) + (
        " (" + ", ".join(parts[1:]) + ")" if len(parts) > 1 else ""
    )

    # Determine status
    if expected:
        expected_t = _parse_version_tuple(expected)
        if installed_t != expected_t:
            return (
                CheckResult("Portal Version", Status.WARN, msg),
                installed,
                expected,
                download_base,
            )

    if latest_portal:
        latest_t = _parse_version_tuple(latest_portal)
        if installed_t < latest_t:
            # Outdated but no specific compatible version required — warn
            return (
                CheckResult(
                    "Portal Version",
                    Status.WARN,
                    msg,
                    detail=f"pip install --upgrade mobilerun, then mobilerun setup",
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
            f"content query --uri {portal_content_uri(LEGACY_PORTAL_PACKAGE_NAME, 'state')}"
        )
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


async def check_tcp(device: AdbDevice, debug: bool) -> CheckResult:
    """Check TCP mode with explicit steps: enable server, forward port, auth, ping."""

    steps = []

    # Step 1: Enable socket server via content provider
    try:
        toggle_uri = portal_content_uri(LEGACY_PORTAL_PACKAGE_NAME, "toggle_socket_server")
        await device.shell(
            f"content insert --uri {toggle_uri} --bind enabled:b:true"
        )
        steps.append("server enabled")
    except Exception as e:
        return CheckResult(
            "TCP Mode",
            Status.FAIL,
            "failed to enable socket server",
            detail=str(e),
        )

    # Wait for server startup
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
                and fwd.remote == f"tcp:{PORTAL_REMOTE_PORT}"
            ):
                match = re.search(r"tcp:(\d+)", fwd.local)
                if match:
                    local_port = int(match.group(1))
                    steps.append(f"reusing forward :{local_port}")
                    break

        if local_port is None:
            local_port = await device.forward_port(PORTAL_REMOTE_PORT)
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
                return CheckResult(
                    "TCP Mode",
                    Status.PASS,
                    f"localhost:{local_port}",
                    detail=", ".join(steps),
                )
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
    device: AdbDevice, use_tcp: bool, debug: bool
) -> tuple[CheckResult, Any]:
    """Fetch full state and verify a11y_tree, phone_state, device_context are present."""
    from mobilerun.tools.android.portal_client import PortalClient

    try:
        portal = PortalClient(device, prefer_tcp=use_tcp)
        await portal.connect()
        method = "tcp" if portal.tcp_available else "content_provider"

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
        ime_id = portal_ime_id(LEGACY_PORTAL_PACKAGE_NAME)
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


# ── Auto-fix Functions ───────────────────────────────────────────


async def fix_enable_accessibility(device: AdbDevice, debug: bool) -> bool:
    """Enable accessibility service. Returns True on success."""
    console.print("  [blue]→ Enabling accessibility service...[/]")
    try:
        await enable_portal_accessibility(device)
        # Verify
        enabled = await check_portal_accessibility(device, debug=debug)
        if enabled:
            console.print("  [green]  accessibility enabled[/]")
            return True
        else:
            console.print(
                "  [yellow]  auto-enable may have failed, opening settings...[/]"
            )
            await device.shell("am start -a android.settings.ACCESSIBILITY_SETTINGS")
            console.print(
                f"  [yellow]  please enable {PORTAL_PACKAGE_NAME} manually[/]"
            )
            return False
    except Exception as e:
        console.print(f"  [red]  failed: {e}[/]")
        try:
            await device.shell("am start -a android.settings.ACCESSIBILITY_SETTINGS")
            console.print(
                f"  [yellow]  please enable {PORTAL_PACKAGE_NAME} manually[/]"
            )
        except Exception:
            pass
        return False


# ── Orchestrator ─────────────────────────────────────────────────


async def run_doctor(
    device_serial: str | None = None,
    debug: bool = False,
) -> None:
    """Run all doctor checks and auto-fix where possible."""
    debug = debug or False

    console.print("\n[bold]Mobilerun Doctor[/]\n")

    results: list[CheckResult] = []

    # 1. SDK Version
    r = await check_sdk_version(debug)
    _print_result(r, debug)
    results.append(r)

    # 2. Config
    r_config, config = await check_config(debug)
    _print_result(r_config, debug)
    results.append(r_config)

    # Load serial from config if not provided
    if device_serial is None and config:
        device_serial = getattr(getattr(config, "device", None), "serial", None)

    # 3. ADB
    r = await check_adb(debug)
    _print_result(r, debug)
    results.append(r)
    if r.status == Status.FAIL:
        _print_summary(results)
        return

    # 5. Device
    r_device, device = await check_device(device_serial, debug)
    _print_result(r_device, debug)
    results.append(r_device)
    if device is None:
        console.print("\n  [dim]Skipping device-dependent checks (no device)[/]")
        _print_summary(results)
        return

    # 6. Portal Installed
    r_portal, is_installed = await check_portal_installed(device, debug)
    _print_result(r_portal, debug)
    if not is_installed:
        # Auto-fix: run setup
        from mobilerun.cli.main import _setup_portal

        await _setup_portal(path=None, device=device.serial, debug=debug)
        # Re-check
        r_portal, is_installed = await check_portal_installed(device, debug)
        if not is_installed:
            results.append(r_portal)
            _print_summary(results)
            return
    results.append(r_portal)

    # 7. Portal Version
    r_version, installed_ver, _, _ = await check_portal_version(device, debug)
    _print_result(r_version, debug)
    if r_version.status == Status.WARN and installed_ver:
        # Auto-fix: run setup to install compatible version
        console.print("  [blue]→ Updating portal...[/]")
        from mobilerun.cli.main import _setup_portal

        await _setup_portal(path=None, device=device.serial, debug=debug)
        # Wait for portal service to initialize after install
        console.print("  [dim]  waiting for portal to start...[/]")
        await asyncio.sleep(3)
        # Re-check
        r_version, _, _, _ = await check_portal_version(device, debug)
        if r_version.status == Status.PASS:
            console.print("  [green]  portal updated[/]")
        else:
            console.print(
                "  [yellow]  update may not have resolved the version mismatch[/]"
            )
    results.append(r_version)

    # 8. Accessibility
    r_a11y, a11y_enabled = await check_accessibility(device, debug)
    if not a11y_enabled:
        _print_result(r_a11y, debug)
        # Auto-fix: enable accessibility
        success = await fix_enable_accessibility(device, debug)
        if success:
            r_a11y = CheckResult("Accessibility", Status.PASS, "enabled (auto-fixed)")
            _print_result(r_a11y, debug)
    else:
        _print_result(r_a11y, debug)
    results.append(r_a11y)

    # 9. Content Provider — connectivity, state, screenshot
    r = await check_content_provider(device, debug)
    _print_result(r, debug)
    results.append(r)

    r_cp_state, cp_portal = await check_portal_state(device, False, debug)
    r_cp_state.name = "State (content)"
    _print_result(r_cp_state, debug)
    results.append(r_cp_state)

    if cp_portal and r_cp_state.status == Status.PASS:
        r = await check_screenshot(cp_portal, debug)
        r.name = "Screenshot (content)"
        _print_result(r, debug)
        results.append(r)

    # 10. TCP — connectivity, state, screenshot
    r = await check_tcp(device, debug)
    _print_result(r, debug)
    results.append(r)

    if r.status == Status.PASS:
        r_tcp_state, tcp_portal = await check_portal_state(device, True, debug)
        r_tcp_state.name = "State (tcp)"
        _print_result(r_tcp_state, debug)
        results.append(r_tcp_state)

        if tcp_portal and r_tcp_state.status == Status.PASS:
            r = await check_screenshot(tcp_portal, debug)
            r.name = "Screenshot (tcp)"
            _print_result(r, debug)
            results.append(r)

    # 12. Keyboard
    r = await check_keyboard(device, debug)
    _print_result(r, debug)
    results.append(r)

    _print_summary(results)


def _print_summary(results: list[CheckResult]) -> None:
    """Print final summary."""
    fails = [r for r in results if r.status == Status.FAIL]
    warns = [r for r in results if r.status == Status.WARN]

    console.print()
    if not fails and not warns:
        console.print("  [bold green]All checks passed.[/]")
    else:
        if fails:
            console.print(f"  [bold red]{len(fails)} issue(s):[/]")
            for r in fails:
                console.print(f"    [red]✗ {r.name}: {r.message}[/]")
                if r.detail:
                    console.print(f"      [dim]{r.detail}[/]")
        if warns:
            console.print(f"  [bold yellow]{len(warns)} warning(s):[/]")
            for r in warns:
                console.print(f"    [yellow]⚠ {r.name}: {r.message}[/]")
                if r.detail:
                    console.print(f"      [dim]{r.detail}[/]")
    console.print()
