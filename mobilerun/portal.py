"""
Portal APK management and device communication utilities.

This module handles downloading, installing, and managing the Mobilerun Portal app
on Android devices. It also provides utilities for checking accessibility service
status and managing device communication modes (TCP and content provider).
"""

import asyncio
import contextlib
import json
import logging
import os
import tempfile
from urllib.parse import urlparse

import requests
from async_adbutils import AdbDevice, adb
from rich.console import Console

from mobilerun import __version__

logger = logging.getLogger("mobilerun")

REPO = "droidrun/mobilerun-portal"
ASSET_NAME = "mobilerun-portal"
DOWNLOAD_BASE = f"https://github.com/{REPO}/releases/download"
GITHUB_API_HOSTS = ["https://api.github.com", "https://ungh.cc"]

VERSION_MAP_GIST_URL = "https://raw.githubusercontent.com/droidrun/gists/refs/heads/main/version_map_android.json"

PORTAL_PACKAGE_NAME = "com.mobilerun.portal"
PORTAL_APK_ASSET_PREFIXES = (
    PORTAL_PACKAGE_NAME,
    "mobilerun-portal-internal",
    ASSET_NAME,
)

# ── Centralized portal identity resolution ──
# ALL portal identifiers (package, a11y service, IME, content URIs) MUST be
# resolved through these helpers. No file should hard-code these strings.

_PORTAL_META = {
    PORTAL_PACKAGE_NAME: {
        "a11y": f"{PORTAL_PACKAGE_NAME}/{PORTAL_PACKAGE_NAME}.service.MobilerunAccessibilityService",
        "ime": f"{PORTAL_PACKAGE_NAME}/.input.MobilerunKeyboardIME",
    },
}

# Artifact channels — mobilerun-portal-internal is a repo/artifact convention,
# not an Android package name.
_ARTIFACT_CHANNELS = {
    PORTAL_PACKAGE_NAME: {
        "repo": "droidrun/mobilerun-portal",
        "asset_name": "mobilerun-portal",
    },
}

A11Y_SERVICE_NAME = _PORTAL_META[PORTAL_PACKAGE_NAME]["a11y"]


def portal_content_uri(pkg: str, path: str) -> str:
    """Build a content URI for the given portal package."""
    return f"content://{pkg}/{path}"


def portal_a11y_service(pkg: str) -> str:
    """Return the accessibility service component name."""
    return _PORTAL_META[pkg]["a11y"]


def portal_ime_id(pkg: str) -> str:
    """Return the IME component name."""
    return _PORTAL_META[pkg]["ime"]


def get_portal_artifact_source(target_package: str) -> dict:
    """Return repo/asset_name for the given portal package."""
    return _ARTIFACT_CHANNELS[target_package]


def get_version_mapping(debug: bool = False) -> dict | None:
    try:
        response = requests.get(VERSION_MAP_GIST_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        if debug:
            print(f"Failed to fetch version mapping: {e}")
        return None


def _version_in_range(version: str, range_str: str) -> bool:
    if "-" not in range_str:
        return False
    try:
        start, end = range_str.split("-", 1)
        v_parts = [int(x) for x in version.split(".")]
        s_parts = [int(x) for x in start.split(".")]
        e_parts = [int(x) for x in end.split(".")]
        return s_parts <= v_parts <= e_parts
    except (ValueError, AttributeError):
        return False


def get_compatible_portal_version(
    mobilerun_version: str, debug: bool = False
) -> tuple[str | None, str, bool]:
    mapping = get_version_mapping(debug)
    if mapping is None:
        return (None, "", False)

    mappings = mapping.get("mappings", {})
    download_base = _normalize_download_base(mapping.get("download_base", DOWNLOAD_BASE))

    # Try exact match first
    if mobilerun_version in mappings:
        return (mappings[mobilerun_version], download_base, True)

    # Try range match (e.g., "0.4.0-0.4.14": "1.0.0")
    for key, portal_version in mappings.items():
        if "-" in key and _version_in_range(mobilerun_version, key):
            return (portal_version, download_base, True)

    return (None, download_base, True)


def _normalize_download_base(download_base: str | None) -> str:
    if not download_base:
        return DOWNLOAD_BASE
    return download_base.replace("droidrun/droidrun-portal", "droidrun/mobilerun-portal")


def _normalize_portal_release_tag(version: str) -> str:
    version = version.strip()
    return version if version.startswith("v") else f"v{version}"


def _extract_release_assets(release: dict) -> list[dict]:
    if "release" in release:
        return release["release"].get("assets", [])
    return release.get("assets", [])


def _asset_download_url(asset: dict) -> str | None:
    return asset.get("browser_download_url") or asset.get("downloadUrl")


def _asset_file_name(asset: dict) -> str:
    name = asset.get("name")
    if name:
        return name

    asset_url = _asset_download_url(asset)
    if not asset_url:
        return ""

    return os.path.basename(urlparse(asset_url).path)


def _is_portal_apk_asset_name(asset_name: str) -> bool:
    lower_name = asset_name.lower()
    if not lower_name.endswith(".apk"):
        return False

    return any(
        lower_name.startswith(prefix.lower()) for prefix in PORTAL_APK_ASSET_PREFIXES
    )


def _portal_apk_asset_priority(asset_name: str) -> tuple[int, str]:
    lower_name = asset_name.lower()
    if "unsigned" in lower_name:
        return (3, lower_name)
    if "release" in lower_name or "stable" in lower_name:
        return (0, lower_name)
    if "debug" in lower_name:
        return (1, lower_name)
    return (2, lower_name)


def _parse_portal_asset_version(asset_name: str) -> str | None:
    stem = os.path.basename(asset_name).removesuffix(".apk")
    lower_stem = stem.lower()

    for suffix in (
        "-release-unsigned",
        "-release-signed",
        "-unsigned",
        "-release",
        "-debug",
        "-stable",
    ):
        if lower_stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            lower_stem = lower_stem[: -len(suffix)]
            break

    for prefix in PORTAL_APK_ASSET_PREFIXES:
        marker = f"{prefix}-"
        if lower_stem.startswith(marker.lower()):
            version = stem[len(marker) :]
            return version.removeprefix("v") or None

    return None


def _format_asset_names(assets: list[dict]) -> str:
    names = [_asset_file_name(asset) or "<unnamed>" for asset in assets]
    return ", ".join(names) if names else "none"


def _select_portal_apk_asset(assets: list[dict]) -> tuple[str, str, str | None]:
    candidates: list[tuple[tuple[int, str], str, str]] = []

    for asset in assets:
        asset_name = _asset_file_name(asset)
        asset_url = _asset_download_url(asset)
        if not asset_name or not asset_url:
            continue
        if not _is_portal_apk_asset_name(asset_name):
            continue
        candidates.append(
            (_portal_apk_asset_priority(asset_name), asset_name, asset_url)
        )

    if not candidates:
        raise Exception(
            "Portal APK asset not found in release. "
            f"Saw assets: {_format_asset_names(assets)}"
        )

    _, asset_name, asset_url = min(candidates, key=lambda candidate: candidate[0])
    return asset_url, asset_name, _parse_portal_asset_version(asset_name)


def _fetch_release_json(release_path: str, debug: bool = False) -> dict:
    path = release_path.lstrip("/")
    response = None
    last_request_error: requests.RequestException | None = None

    for host in GITHUB_API_HOSTS:
        url = f"{host}/repos/{REPO}/{path}"
        try:
            response = requests.get(url)
        except requests.RequestException as e:
            last_request_error = e
            if debug:
                print(f"Failed to fetch release from {host}: {e}")
            continue

        if response.status_code == 200:
            if debug:
                print(f"Using GitHub release on {host}")
            return response.json()

    if response is not None:
        response.raise_for_status()

    if last_request_error is not None:
        raise Exception(
            "Failed to fetch Portal release metadata from all configured hosts"
        ) from last_request_error

    raise Exception("No GitHub API hosts configured")


def _get_release_assets_by_tag(version: str, debug: bool = False) -> list[dict]:
    tag = _normalize_portal_release_tag(version)
    release = _fetch_release_json(f"releases/tags/{tag}", debug)
    return _extract_release_assets(release)


def _resolve_versioned_portal_apk_asset(
    version: str, download_base: str, debug: bool = False
) -> tuple[str, str, str]:
    tag = _normalize_portal_release_tag(version)

    try:
        assets = _get_release_assets_by_tag(tag, debug)
        asset_url, asset_name, asset_version = _select_portal_apk_asset(assets)
        return asset_url, asset_version or tag.removeprefix("v"), asset_name
    except Exception as e:
        if debug:
            print(f"Failed to resolve release assets for {tag}, using fallback URL: {e}")

    asset_name = f"{PORTAL_PACKAGE_NAME}-{tag.removeprefix('v')}-debug.apk"
    base = _normalize_download_base(download_base).rstrip("/")
    return f"{base}/{tag}/{asset_name}", tag.removeprefix("v"), asset_name


def _resolve_latest_portal_apk_asset(debug: bool = False) -> tuple[str, str, str]:
    assets = get_latest_release_assets(debug)
    asset_url, asset_name, asset_version = _select_portal_apk_asset(assets)
    return asset_url, asset_version or "unknown", asset_name


@contextlib.contextmanager
def download_versioned_portal_apk(
    version: str, download_base: str, debug: bool = False
):
    """Download a specific Portal APK version."""
    console = Console()
    asset_url, asset_version, _ = _resolve_versioned_portal_apk_asset(
        version, download_base, debug
    )

    console.print(f"Downloading Portal APK [bold]{asset_version}[/bold]")
    if debug:
        console.print(f"Asset URL: {asset_url}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".apk")
    try:
        r = requests.get(asset_url, stream=True)
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp.close()
        yield tmp.name
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def get_latest_release_assets(debug: bool = False):
    """
    Fetch the latest Portal APK release assets from GitHub.

    Args:
        debug: Enable debug logging

    Returns:
        List of asset dictionaries from the latest GitHub release

    Raises:
        requests.HTTPError: If the GitHub API request fails
    """
    latest_release = _fetch_release_json("releases/latest", debug)
    return _extract_release_assets(latest_release)


@contextlib.contextmanager
def download_portal_apk(debug: bool = False):
    """
    Download the latest Portal APK from GitHub releases.

    This context manager downloads the APK to a temporary file and yields
    the file path. The file is automatically deleted when the context exits.

    Args:
        debug: Enable debug logging

    Yields:
        str: Path to the downloaded APK file

    Raises:
        Exception: If the Portal APK asset is not found in the release
        requests.HTTPError: If the download fails
    """
    console = Console()
    asset_url, asset_version, _ = _resolve_latest_portal_apk_asset(debug)

    console.print(f"Found Portal APK [bold]{asset_version}[/bold]")
    if debug:
        console.print(f"Asset URL: {asset_url}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".apk")
    try:
        r = requests.get(asset_url, stream=True)
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp.close()
        yield tmp.name
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


async def enable_portal_accessibility(
    device: AdbDevice, service_name: str = A11Y_SERVICE_NAME
):
    """
    Enable the Portal accessibility service on the device.

    Args:
        device: ADB device connection
        service_name: Full accessibility service name (default: Portal service)

    Note:
        This may fail on some devices due to security restrictions.
        Manual enablement may be required.
    """
    await device.shell(
        f"settings put secure enabled_accessibility_services {service_name}"
    )
    await device.shell("settings put secure accessibility_enabled 1")


async def check_portal_accessibility(
    device: AdbDevice, service_name: str = A11Y_SERVICE_NAME, debug: bool = False
) -> bool:
    """
    Check if the Portal accessibility service is enabled.

    Args:
        device: ADB device connection
        service_name: Full accessibility service name to check
        debug: Enable debug logging

    Returns:
        True if the accessibility service is enabled, False otherwise
    """
    a11y_services = await device.shell(
        "settings get secure enabled_accessibility_services"
    )
    if service_name not in a11y_services:
        if debug:
            print(a11y_services)
        return False

    a11y_enabled = await device.shell("settings get secure accessibility_enabled")
    if a11y_enabled != "1":
        if debug:
            print(a11y_enabled)
        return False

    return True


async def ping_portal(device: AdbDevice, debug: bool = False):
    """
    Ping the Mobilerun Portal to check if it is installed and accessible.
    """
    try:
        packages = await device.list_packages()
    except Exception as e:
        raise Exception("Failed to list packages") from e

    if PORTAL_PACKAGE_NAME not in packages:
        if debug:
            print(packages)
        raise Exception("Portal is not installed on the device")

    if not await check_portal_accessibility(device, debug=debug):
        await device.shell("am start -a android.settings.ACCESSIBILITY_SETTINGS")
        raise Exception(
            "Mobilerun Portal is not enabled as an accessibility service on the device"
        )


async def ping_portal_content(device: AdbDevice, debug: bool = False):
    """
    Test Portal accessibility via content provider.

    Args:
        device: ADB device connection
        debug: Enable debug logging

    Raises:
        Exception: If Portal is not reachable via content provider
    """
    try:
        uri = portal_content_uri(PORTAL_PACKAGE_NAME, "state")
        state = await device.shell(
            f"content query --uri {uri}"
        )
        if "Row: 0 result=" not in state:
            raise Exception("Failed to get state from Mobilerun Portal")
    except Exception as e:
        raise Exception("Mobilerun Portal is not reachable") from e


async def ping_portal_tcp(device: AdbDevice, debug: bool = False):
    """
    Test Portal accessibility via TCP mode.

    Args:
        device: ADB device connection
        debug: Enable debug logging

    Raises:
        Exception: If Portal is not reachable via TCP or port forwarding fails
    """
    from mobilerun.tools.driver.android import AndroidDriver

    try:
        driver = AndroidDriver(serial=device.serial, use_tcp=True)
        await driver.connect()
    except Exception as e:
        raise Exception("Failed to setup TCP forwarding") from e


async def set_overlay_offset(device: AdbDevice, offset: int):
    """
    Set the overlay offset using the /overlay_offset portal content provider endpoint.
    """
    try:
        uri = portal_content_uri(PORTAL_PACKAGE_NAME, "overlay_offset")
        cmd = f'content insert --uri "{uri}" --bind offset:i:{offset}'
        await device.shell(cmd)
    except Exception as e:
        raise Exception("Error setting overlay offset") from e


async def toggle_overlay(device: AdbDevice, visible: bool):
    """Toggle the overlay visibility.

    Args:
        device: Device to toggle the overlay on
        visible: Whether to show the overlay

    throws:
        Exception: If the overlay toggle fails
    """
    try:
        visible_str = "true" if visible else "false"
        uri = portal_content_uri(PORTAL_PACKAGE_NAME, "overlay_visible")
        cmd = f'content insert --uri "{uri}" --bind visible:b:{visible_str}'
        await device.shell(cmd)
    except Exception as e:
        raise Exception("Failed to toggle overlay") from e


async def setup_keyboard(device: AdbDevice):
    """
    Set up the Mobilerun keyboard as the default input method.
    Simple setup that just switches to Mobilerun keyboard without saving/restoring.

    throws:
        Exception: If the keyboard setup fails
    """
    try:
        ime = portal_ime_id(PORTAL_PACKAGE_NAME)
        await device.shell(f"ime enable {ime}")
        await device.shell(f"ime set {ime}")
    except Exception as e:
        raise Exception("Error setting up keyboard") from e


async def disable_keyboard(
    device: AdbDevice,
    target_ime: str | None = None,
):
    """
    Disable a specific IME (keyboard) and optionally switch to another.
    By default, disables the Mobilerun keyboard.

    Args:
        target_ime: The IME package/activity to disable (default: Mobilerun keyboard)

    Returns:
        bool: True if disabled successfully, False otherwise
    """
    if target_ime is None:
        target_ime = portal_ime_id(PORTAL_PACKAGE_NAME)
    try:
        await device.shell(f"ime disable {target_ime}")
        return True
    except Exception as e:
        raise Exception("Error disabling keyboard") from e


async def setup_portal(
    device: AdbDevice,
    debug: bool = False,
) -> bool:
    """Download, install, and enable the Portal APK on a device.

    Uses version mapping to find the compatible Portal version for the
    current mobilerun SDK version.  Falls back to the latest release if
    the mapping is unavailable.

    Args:
        device: ADB device connection.
        debug: Enable debug logging.

    Returns:
        True if setup completed successfully, False otherwise.
    """
    try:
        portal_version, download_base, mapping_fetched = get_compatible_portal_version(
            __version__, debug
        )

        if portal_version:
            apk_context = download_versioned_portal_apk(
                portal_version, download_base, debug
            )
        else:
            if not mapping_fetched:
                logger.warning(
                    "Could not fetch version mapping, falling back to latest portal"
                )
            apk_context = download_portal_apk(debug)

        with apk_context as apk_path:
            if not os.path.exists(apk_path):
                logger.error(f"APK file not found at {apk_path}")
                return False

            logger.info("Installing Portal APK...")
            try:
                await device.install(
                    apk_path, uninstall=True, flags=["-g"], silent=not debug
                )
            except Exception as e:
                logger.error(f"Portal installation failed: {e}")
                return False

            logger.info("Portal APK installed")

            try:
                await enable_portal_accessibility(device)
                # Wait for the service to become responsive
                await _wait_for_portal_service(device)
                logger.info("Accessibility service enabled")
            except Exception as e:
                logger.warning(f"Could not auto-enable accessibility service: {e}")
                try:
                    await device.shell(
                        "am start -a android.settings.ACCESSIBILITY_SETTINGS"
                    )
                except Exception:
                    pass
                return False

        return True

    except Exception as e:
        logger.error(f"Portal setup failed: {e}")
        if debug:
            import traceback

            logger.debug(traceback.format_exc())
        return False


async def _wait_for_portal_service(
    device: AdbDevice, timeout: float = 10.0, interval: float = 1.0
) -> None:
    """Poll the content provider until the accessibility service is responsive.

    Uses the simple ``/state`` endpoint which responds as soon as the
    service process is alive, without requiring an active window.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            uri = portal_content_uri(PORTAL_PACKAGE_NAME, "state")
            state = await device.shell(
                f"content query --uri {uri}"
            )
            if '"status":"success"' in state:
                return
        except Exception:
            pass
        await asyncio.sleep(interval)
    logger.warning("Portal service did not become responsive within timeout")


def _parse_portal_version(raw_output: str) -> str | None:
    """Extract portal version string from content provider output."""
    try:
        if "result=" in raw_output:
            json_str = raw_output.split("result=", 1)[1].strip()
            data = json.loads(json_str)
            if data.get("status") == "success":
                return data.get("result") or data.get("data")
    except Exception:
        pass
    return None


async def ensure_portal_ready(
    device: AdbDevice,
    debug: bool = False,
) -> None:
    """Run parallel health checks and auto-fix portal issues.

    Performs three checks concurrently:
    1. Is the Portal APK installed?
    2. Is the installed version compatible?
    3. Is the accessibility service enabled?

    If any check fails, attempts to fix automatically (install/upgrade
    APK, enable accessibility).  Raises on unrecoverable failure.

    Args:
        device: ADB device connection.
        debug: Enable debug logging.

    Raises:
        RuntimeError: If portal cannot be made ready after auto-fix.
    """
    # ── parallel checks ──────────────────────────────────────────
    packages_task = device.list_packages()
    version_task = device.shell(
        f"content query --uri {portal_content_uri(PORTAL_PACKAGE_NAME, 'version')}"
    )
    a11y_task = device.shell("settings get secure enabled_accessibility_services")

    packages, version_raw, a11y_services = await asyncio.gather(
        packages_task, version_task, a11y_task, return_exceptions=True
    )

    # If all checks failed, the device is likely unreachable — skip
    # auto-setup and let AndroidDriver.connect() surface the real error.
    if (
        isinstance(packages, Exception)
        and isinstance(version_raw, Exception)
        and isinstance(a11y_services, Exception)
    ):
        logger.debug(f"Portal health check skipped (device unreachable): {packages}")
        return

    # ── evaluate results ─────────────────────────────────────────
    is_installed = isinstance(packages, list) and PORTAL_PACKAGE_NAME in packages

    installed_version = (
        _parse_portal_version(version_raw) if isinstance(version_raw, str) else None
    )

    a11y_enabled = isinstance(a11y_services, str) and A11Y_SERVICE_NAME in a11y_services

    # Check version compatibility
    needs_upgrade = False
    if is_installed and installed_version:
        expected_version, _, mapping_fetched = get_compatible_portal_version(
            __version__, debug
        )
        if expected_version and mapping_fetched:
            needs_upgrade = installed_version != expected_version.lstrip("v")
            if needs_upgrade:
                logger.info(
                    f"Portal version mismatch: installed={installed_version}, "
                    f"expected={expected_version}"
                )

    # ── fix if needed ────────────────────────────────────────────
    if not is_installed or needs_upgrade:
        reason = "not installed" if not is_installed else "outdated"
        logger.info(f"Portal {reason}, running auto-setup...")
        success = await setup_portal(device, debug)
        if not success:
            raise RuntimeError(
                f"Portal auto-setup failed ({reason}). "
                "Run 'mobilerun doctor' for diagnostics."
            )
        # After install, accessibility is already enabled by setup_portal
        return

    if not a11y_enabled:
        logger.info("Portal accessibility service not enabled, enabling...")
        try:
            await enable_portal_accessibility(device)
            # Verify settings were applied
            if not await check_portal_accessibility(device, debug=debug):
                raise RuntimeError(
                    "Could not enable Portal accessibility service. "
                    "Please enable it manually in device settings, "
                    "or run 'mobilerun setup'."
                )
            # Wait for the service process to start and become responsive
            await _wait_for_portal_service(device)
            logger.info("Accessibility service enabled")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to enable accessibility service: {e}. "
                "Run 'mobilerun doctor' for diagnostics."
            ) from e


async def test():
    device = await adb.device()
    await ping_portal(device, debug=False)


if __name__ == "__main__":
    asyncio.run(test())
