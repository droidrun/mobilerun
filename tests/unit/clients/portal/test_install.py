"""Offline contracts for Portal installs, setup options, and release assets."""

import asyncio

import pytest
from mobilerun_core_local.driver.android import portal as sdk_portal

import mobilerun
import mobilerun.clients.portal.install as portal
from mobilerun.clients.portal.setup import portal_version_kwargs


def test_portal_wait_timeout_without_polling():
    from mobilerun.clients.portal.install import wait_for_portal_service

    assert asyncio.run(wait_for_portal_service(object(), timeout=0)) is False


def test_install_portal_with_apk_returns_result(tmp_path, monkeypatch):
    from mobilerun.clients.portal.install import install_portal

    apk = tmp_path / "portal.apk"
    apk.write_bytes(b"apk")

    class Device:
        async def install(self, *args, **kwargs):
            pass

        async def shell(self, command):
            return '"status":"success"'

    async def ready(*args, **kwargs):
        return True

    monkeypatch.setattr(portal, "enable_portal_accessibility", ready)
    monkeypatch.setattr(portal, "wait_for_portal_service", ready)

    result = asyncio.run(install_portal(Device(), apk_path=str(apk)))
    assert result.installed
    assert result.apk_source == str(apk)


def test_install_portal_missing_apk_raises(tmp_path):
    from mobilerun.clients.portal.install import install_portal
    from mobilerun.errors import PortalApkNotFound

    with pytest.raises(PortalApkNotFound):
        asyncio.run(install_portal(object(), apk_path=str(tmp_path / "missing.apk")))


def test_install_portal_accessibility_fallback_requests_manual_setup(tmp_path):
    from mobilerun.clients.portal.install import install_portal

    apk = tmp_path / "portal.apk"
    apk.write_bytes(b"apk")

    class Device:
        async def install(self, *args, **kwargs):
            pass

        async def shell(self, command):
            return "opened"

    result = asyncio.run(install_portal(Device(), apk_path=str(apk), wait_timeout=0))
    assert result.installed
    assert result.needs_manual_accessibility


def test_setup_cli_prints_install_success_messages(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from mobilerun.cli.devices import setup as setup_module
    from mobilerun.clients.portal.install import PortalSetupResult

    class Device:
        pass

    async def get_device(serial):
        return Device()

    async def install(device, **kwargs):
        return PortalSetupResult(True, True, False, str(tmp_path / "portal.apk"))

    monkeypatch.setattr(setup_module.adb, "device", get_device)
    monkeypatch.setattr(setup_module, "install_portal", install)
    result = CliRunner().invoke(
        setup_module.setup, ["test-device", "--android", "--path", "portal.apk"]
    )
    assert result.exit_code == 0
    assert "Using provided APK: portal.apk" in result.output
    assert "Step 1/2: Installing APK:" in result.output
    assert "Installation successful!" in result.output
    assert "Accessibility service enabled successfully!" in result.output
    assert "Setup complete!" in result.output


def test_wait_for_portal_logs_retryable_failure(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from mobilerun.clients.portal.install import logger, wait_for_portal_service

    warning = Mock()
    monkeypatch.setattr(logger, "warning", warning)
    device = SimpleNamespace(shell=AsyncMock(side_effect=OSError("temporary")))
    assert not asyncio.run(
        wait_for_portal_service(device, timeout=0.001, interval=0.002)
    )
    warning.assert_called_once()


def test_portal_setup_can_ask_for_an_in_place_update():
    def new_setup(device, debug=False, *, version=None, uninstall=True):
        return None

    def version_only_setup(device, debug=False, version=None):
        return None

    def old_setup(device, debug=False):
        return None

    assert portal_version_kwargs(new_setup, in_place=True) == {
        "version": mobilerun.__version__,
        "uninstall": False,
    }
    assert portal_version_kwargs(new_setup) == {"version": mobilerun.__version__}
    assert portal_version_kwargs(version_only_setup) == {
        "version": mobilerun.__version__
    }
    assert portal_version_kwargs(old_setup) == {}
    assert portal_version_kwargs(old_setup, in_place=True) == {}


def test_sdk_version_check_still_flags_an_rc_after_its_release(monkeypatch):
    from mobilerun.diagnostics import checks as doctor
    from mobilerun.diagnostics import versions as sdk_releases

    class Response:
        status_code = 200

        def json(self):
            return {"tag_name": "v0.6.22"}

    monkeypatch.setattr(doctor, "__version__", "0.6.22rc1")
    monkeypatch.setattr(sdk_releases.requests, "get", lambda url, **kwargs: Response())

    result = asyncio.run(doctor.check_sdk_version(debug=False))

    assert result.status == doctor.Status.WARN


SIGNED = "com.mobilerun.portal-0.7.1.apk"
DEBUG = "com.mobilerun.portal-0.7.1-debug.apk"
UNSIGNED = "com.mobilerun.portal-0.7.1-release-unsigned.apk"
PUBLIC = "mobilerun-portal-v0.7.1-release.apk"
INTERNAL = "mobilerun-portal-internal-v0.7.1-debug.apk"
DOWNLOAD_BASE = "https://github.com/droidrun/mobilerun-portal/releases/download"
STALE_BASE = "https://github.com/droidrun/droidrun-portal/releases/download"


def _github_asset(name):
    return {
        "name": name,
        "browser_download_url": f"https://example.com/releases/{name}",
    }


@pytest.mark.parametrize(
    "names,expected",
    [
        ([UNSIGNED, DEBUG], DEBUG),
        ([UNSIGNED, DEBUG, SIGNED], SIGNED),
        ([PUBLIC], PUBLIC),
        ([INTERNAL], INTERNAL),
    ],
    ids=["debug-over-unsigned", "signed-over-debug", "public-name", "internal-name"],
)
def test_selects_release_asset(names, expected):
    url, name, version = sdk_portal._select_portal_apk_asset(
        [_github_asset(n) for n in names]
    )
    assert name == expected and url.endswith("/" + expected)
    assert version == "0.7.1"


def test_legacy_droidrun_portal_asset_is_not_active_match():
    with pytest.raises(Exception, match="Portal APK asset not found"):
        sdk_portal._select_portal_apk_asset(
            [_github_asset("droidrun-portal-v0.4.6.apk")]
        )


def test_ungh_download_url_asset_matches_without_name():
    url = "https://example.com/releases/download/v0.7.1/mobilerun-portal-v0.7.1.apk"
    assert sdk_portal._select_portal_apk_asset([{"downloadUrl": url}]) == (
        url,
        "mobilerun-portal-v0.7.1.apk",
        "0.7.1",
    )


def test_no_matching_apk_reports_seen_assets():
    names = ["release-notes.txt", "other-app-1.0.0.apk"]
    with pytest.raises(Exception, match="Portal APK asset not found") as exc:
        sdk_portal._select_portal_apk_asset([_github_asset(name) for name in names])
    assert all(name in str(exc.value) for name in names)


@pytest.mark.parametrize("tag", [None, "v0.7.1"], ids=["latest", "tag"])
def test_release_lookup_retries_next_host_on_request_error(monkeypatch, tag):
    calls = []
    release = {"assets": [_github_asset(DEBUG)]}

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"release": release} if tag else release

    def get(url, **kwargs):
        calls.append(url)
        if "api.github.com" in url:
            raise sdk_portal.requests.ConnectionError("offline")
        return Response()

    monkeypatch.setattr(sdk_portal.requests, "get", get)
    assets = (
        sdk_portal._get_release_assets_by_tag(tag)
        if tag
        else sdk_portal.get_latest_release_assets()
    )
    assert len(calls) == 2
    assert "api.github.com" in calls[0] and "ungh.cc" in calls[1]
    assert assets[0]["name"] == DEBUG


def test_release_lookup_raises_clear_error_when_all_hosts_fail(monkeypatch):
    def get(url, **kwargs):
        raise sdk_portal.requests.ConnectionError("offline")

    monkeypatch.setattr(sdk_portal.requests, "get", get)
    with pytest.raises(Exception, match="all configured hosts"):
        sdk_portal.get_latest_release_assets()


def test_versioned_resolution_uses_release_tag_assets(monkeypatch):
    def assets(version, debug=False):
        return [_github_asset(n) for n in [UNSIGNED, DEBUG, SIGNED]]

    monkeypatch.setattr(sdk_portal, "_get_release_assets_by_tag", assets)
    url, version, name = sdk_portal._resolve_versioned_portal_apk_asset(
        "0.7.1", DOWNLOAD_BASE
    )
    assert (url, version, name) == (
        _github_asset(SIGNED)["browser_download_url"],
        "0.7.1",
        SIGNED,
    )


def test_versioned_resolution_normalizes_stale_download_base_fallback(monkeypatch):
    def offline(version, debug=False):
        raise RuntimeError("offline")

    monkeypatch.setattr(sdk_portal, "_get_release_assets_by_tag", offline)
    assert sdk_portal._resolve_versioned_portal_apk_asset("v0.7.1", STALE_BASE) == (
        f"{DOWNLOAD_BASE}/v0.7.1/{SIGNED}",
        "0.7.1",
        SIGNED,
    )
    assert sdk_portal._normalize_download_base(STALE_BASE) == DOWNLOAD_BASE


@pytest.mark.parametrize("name", [SIGNED, DEBUG, UNSIGNED, PUBLIC, INTERNAL])
def test_asset_version_parsing_ignores_build_suffixes(name):
    assert sdk_portal._parse_portal_asset_version(name) == "0.7.1"
