import importlib
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.diagnostics.results import CheckResult, Status


@pytest.fixture
def checks(monkeypatch):
    module = importlib.import_module("mobilerun.cli.doctor")
    ok = CheckResult("test", Status.PASS, "ready")
    fail = CheckResult("Portal", Status.FAIL, "not installed")
    warn = CheckResult("TCP", Status.WARN, "not configured")
    for name in (
        "check_sdk_version",
        "check_adb",
        "check_content_provider",
        "check_keyboard",
    ):
        monkeypatch.setattr(module, name, AsyncMock(return_value=ok))
    monkeypatch.setattr(module, "check_config", AsyncMock(return_value=(ok, None)))
    monkeypatch.setattr(
        module, "check_device", AsyncMock(return_value=(ok, NS(serial="serial")))
    )
    monkeypatch.setattr(
        module, "check_portal_installed", AsyncMock(return_value=(fail, False))
    )
    monkeypatch.setattr(
        module, "check_portal_version", AsyncMock(return_value=(ok, "1", "1", None))
    )
    monkeypatch.setattr(
        module, "check_accessibility", AsyncMock(return_value=(ok, True))
    )
    monkeypatch.setattr(
        module, "check_portal_state", AsyncMock(return_value=(ok, None))
    )
    monkeypatch.setattr(module, "check_tcp", AsyncMock(return_value=warn))
    monkeypatch.setattr(module, "install_portal", AsyncMock())
    monkeypatch.setattr(
        module, "fix_enable_accessibility", AsyncMock(return_value=True)
    )
    return module, ok, fail


def test_doctor_default_does_not_repair_missing_portal(checks):
    module, _, _ = checks
    result = CliRunner().invoke(cli, ["doctor", "-d", "serial"])
    assert result.exit_code == 1, result.output
    module.install_portal.assert_not_awaited()
    module.fix_enable_accessibility.assert_not_awaited()


def test_doctor_fix_installs_and_rechecks(checks):
    module, ok, fail = checks
    module.check_portal_installed.side_effect = [(fail, False), (ok, True)]
    result = CliRunner().invoke(cli, ["doctor", "-d", "serial", "--fix"])
    assert result.exit_code == 0, (result.output, result.exception)
    module.install_portal.assert_awaited_once()
    assert module.check_portal_installed.await_count == 2
    assert "warning" in result.output


@pytest.mark.parametrize("fix", [False, True])
def test_doctor_accessibility_repairs_only_when_requested(checks, fix):
    module, ok, _ = checks
    module.check_portal_installed.return_value = (ok, True)
    module.check_accessibility.return_value = (
        CheckResult("Accessibility", Status.FAIL, "disabled"),
        False,
    )
    result = CliRunner().invoke(
        cli, ["doctor", "-d", "serial", *(["--fix"] if fix else [])]
    )
    assert result.exit_code == (0 if fix else 1), result.output
    assert module.fix_enable_accessibility.await_count == int(fix)


def test_doctor_adb_failure_exits_one(checks):
    module, _, fail = checks
    module.check_adb.return_value = fail
    result = CliRunner().invoke(cli, ["doctor", "-d", "serial"])
    assert result.exit_code == 1
    module.check_device.assert_not_awaited()
    module.install_portal.assert_not_awaited()


def test_doctor_menu_offers_diagnose_and_repair(checks, monkeypatch):
    module, _, _ = checks
    monkeypatch.setattr(module, "is_interactive", lambda: True)
    seen = []

    def choose(label, choices):
        seen.extend(choice.label for choice in choices)
        return None

    monkeypatch.setattr(module, "select_prompt", choose)
    result = CliRunner().invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert seen == ["Diagnose", "Diagnose and repair"]


def test_setup_failure_exits_one(monkeypatch):
    module = importlib.import_module("mobilerun.cli.devices.setup")
    monkeypatch.setattr(module.adb, "device", AsyncMock(return_value=None))
    result = CliRunner().invoke(cli, ["device", "setup", "-d", "serial"])
    assert result.exit_code == 1, result.output
    assert "Could not get device object" in result.output


def test_tcp_read_only_does_not_toggle_or_forward(monkeypatch):
    import asyncio
    from mobilerun.diagnostics import checks

    class Phone:
        serial = "serial"
        shell = AsyncMock()
        forward_port = AsyncMock()

        async def forward_list(self):
            if False:
                yield None

    phone = Phone()
    monkeypatch.setattr(
        checks.PortalClient, "_fetch_auth_token", AsyncMock(return_value=None)
    )
    result = asyncio.run(checks.check_tcp(phone, False))
    assert result.status == Status.WARN
    assert "--fix" in result.message
    phone.shell.assert_not_awaited()
    phone.forward_port.assert_not_awaited()


def test_setup_manual_accessibility_is_not_success(monkeypatch):
    from mobilerun.clients.portal.install import PortalSetupResult

    module = importlib.import_module("mobilerun.cli.devices.setup")
    monkeypatch.setattr(module.adb, "device", AsyncMock(return_value=object()))
    monkeypatch.setattr(
        module,
        "install_portal",
        AsyncMock(return_value=PortalSetupResult(True, False, True)),
    )
    result = CliRunner().invoke(cli, ["device", "setup", "-d", "serial"])
    assert result.exit_code == 1, result.output
    assert "manually enable" in result.output
    assert "Setup complete!" not in result.output
