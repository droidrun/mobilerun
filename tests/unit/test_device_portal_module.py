import asyncio

import pytest

import mobilerun.clients.portal.install as portal


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
    import asyncio
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
