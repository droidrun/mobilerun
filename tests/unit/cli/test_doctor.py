"""Doctor diagnostics, opt-in repairs, and read-only Portal transport checks."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.cli.doctor import doctor
from mobilerun.diagnostics import checks as diagnostics
from mobilerun.diagnostics.checks import ReadyTcp
from mobilerun.diagnostics.results import CheckResult, Status
from tests.support.doctor import checks as checks

doctor_module = importlib.import_module("mobilerun.cli.doctor")


@pytest.mark.parametrize("flag", ["--device", "-d"])
def test_doctor_legacy_device_option(monkeypatch, flag):
    run = AsyncMock()
    monkeypatch.setattr(doctor_module, "run_doctor", run)
    result = CliRunner().invoke(doctor, [flag, "emulator-5554"])
    assert result.exit_code == 0, result.output
    run.assert_awaited_once_with("emulator-5554", False, fix=False)


def test_doctor_passes_flags_to_diagnostics(monkeypatch):
    run = AsyncMock()
    monkeypatch.setattr(doctor_module, "run_doctor", run)
    result = CliRunner().invoke(doctor, ["emulator-5554", "--android", "--debug"])
    assert result.exit_code == 0, result.output
    run.assert_awaited_once_with("emulator-5554", True, fix=False)


def test_doctor_portal_http_android_runs_local_only(monkeypatch):
    run = AsyncMock()
    monkeypatch.setattr(doctor_module, "run_doctor", run)
    result = CliRunner().invoke(doctor, ["http://127.0.0.1:8080", "-a"])
    assert result.exit_code == 0, result.output
    assert " ".join(result.output.split()) == (
        "! doctor: device checks for Portal HTTP targets aren't supported yet; ran local checks"
    )
    run.assert_awaited_once_with(None, False, fix=False)


def test_doctor_without_target_uses_discovery(monkeypatch):
    run = AsyncMock()
    monkeypatch.setattr(doctor_module, "run_doctor", run)
    from mobilerun.clients.portal import discovery

    monkeypatch.setattr(
        discovery.adb,
        "list",
        AsyncMock(return_value=[SimpleNamespace(serial="test-device", state="device")]),
    )
    result = CliRunner().invoke(doctor, [])
    assert result.exit_code == 0, result.output
    assert "Using test-device" in result.stderr
    run.assert_awaited_once_with("test-device", False, fix=False)


def test_doctor_rejects_multiple_targets():
    result = CliRunner().invoke(doctor, ["serial", "--android", "--ios"])
    assert result.exit_code == 2
    assert "Choose at most one device type flag" in result.output


@pytest.mark.parametrize("kind", ["ios", "cloud", "remote"])
def test_non_android_doctor_runs_local_checks_only(monkeypatch, kind):
    run = AsyncMock()
    monkeypatch.setattr(doctor_module, "run_doctor", run)
    result = CliRunner().invoke(doctor, ["target", f"--{kind}"])
    assert result.exit_code == 0, result.output
    assert (
        "doctor: device checks support Android only; ran local checks" in result.output
    )
    run.assert_awaited_once_with(None, False, fix=False)


def test_tcp_check_reuses_existing_forward(monkeypatch):
    class Device:
        serial = "serial"
        shell = AsyncMock()

        async def forward_list(self):
            yield SimpleNamespace(serial="serial", remote="tcp:8080", local="tcp:1234")

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, **kwargs):
            assert url == "http://localhost:1234/ping"
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(diagnostics.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(
        diagnostics.PortalClient, "_fetch_auth_token", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(diagnostics.httpx, "AsyncClient", Client)
    result = asyncio.run(diagnostics.check_tcp(Device(), False))
    assert result.status == diagnostics.Status.PASS, result.message


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
    class Phone:
        serial = "serial"
        shell = AsyncMock()
        forward_port = AsyncMock()

        async def forward_list(self):
            if False:
                yield None

    phone = Phone()
    monkeypatch.setattr(
        diagnostics.PortalClient, "_fetch_auth_token", AsyncMock(return_value=None)
    )
    result = asyncio.run(diagnostics.check_tcp(phone, False))
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


@pytest.mark.parametrize("lost_forward", [True, False])
def test_plain_doctor_uses_only_existing_tcp_connection(
    checks, monkeypatch, lost_forward
):

    module, ok, _ = checks
    device = module.check_device.return_value[1]

    async def forwards():
        yield SimpleNamespace(serial=device.serial, local="tcp:1234", remote="tcp:8080")

    device.forward_list = forwards
    device.forward_port = AsyncMock()
    device.shell = AsyncMock()
    connect = AsyncMock(side_effect=AssertionError("must not repair"))
    monkeypatch.setattr(diagnostics.PortalClient, "connect", connect)
    monkeypatch.setattr(
        diagnostics.PortalClient,
        "_fetch_auth_token",
        AsyncMock(return_value="test-token"),
    )
    paths = []

    def respond(request):
        assert request.url.port == 1234
        assert request.headers["Authorization"] == "Bearer test-token"
        paths.append(request.url.path)
        if request.url.path == "/ping":
            return httpx.Response(200)
        if lost_forward:
            raise httpx.ConnectError("forward disappeared")
        if request.url.path == "/state_full":
            return httpx.Response(
                200,
                json={
                    "result": dict.fromkeys(
                        ("a11y_tree", "phone_state", "device_context")
                    )
                },
            )
        assert request.url.path == "/screenshot"
        return httpx.Response(200, json={"status": "success", "result": "cG5n"})

    client = httpx.AsyncClient
    monkeypatch.setattr(
        diagnostics.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(respond), **kw),
    )

    async def state(device, debug, **kwargs):
        if not kwargs:
            return ok, None
        return await diagnostics.check_portal_state(device, debug, **kwargs)

    module.check_portal_state.side_effect = state
    module.check_portal_installed.return_value = (ok, True)
    monkeypatch.setattr(module, "check_tcp", diagnostics.check_tcp)
    result = CliRunner().invoke(module.doctor, ["-d", device.serial])
    assert result.exit_code == int(lost_forward), result.output
    assert paths == ["/ping", "/state_full", *([] if lost_forward else ["/screenshot"])]
    connect.assert_not_awaited()
    device.forward_port.assert_not_awaited()
    device.shell.assert_not_awaited()


def test_plain_doctor_reports_broken_tcp_screenshot(checks, monkeypatch):
    module, ok, _ = checks
    tcp_portal = object()
    module.check_portal_installed.return_value = (ok, True)
    module.check_tcp.return_value = ReadyTcp(1234, {}, "ping ok")
    module.check_portal_state.side_effect = [(ok, None), (ok, tcp_portal)]
    screenshot = AsyncMock(
        return_value=CheckResult("Screenshot", Status.FAIL, "empty response")
    )
    monkeypatch.setattr(module, "check_screenshot", screenshot)
    result = CliRunner().invoke(module.doctor, ["-d", "serial"])
    assert result.exit_code == 1, result.output
    assert "Screenshot (tcp): empty response" in result.output
    screenshot.assert_awaited_once_with(tcp_portal, False)
    module.check_tcp.assert_awaited_once_with(
        module.check_device.return_value[1], False, fix=False
    )


def test_device_without_serial_uses_adb_default(monkeypatch):
    device = SimpleNamespace(
        serial="device-1", get_state=AsyncMock(return_value="device")
    )
    lookup = AsyncMock(return_value=device)
    monkeypatch.setattr(diagnostics.adb, "device", lookup)
    result, selected = asyncio.run(diagnostics.check_device(None, False))
    assert result.status == Status.PASS
    assert selected is device
    lookup.assert_awaited_once_with()


@pytest.mark.parametrize(
    "check, status",
    [
        (diagnostics.check_content_provider, Status.FAIL),
        (diagnostics.check_keyboard, Status.WARN),
    ],
)
def test_non_text_adb_response_is_reported_as_failure(check, status):
    result = asyncio.run(
        check(SimpleNamespace(shell=AsyncMock(return_value=b"unexpected")), True)
    )
    assert result.status == status
    assert "non-text output" in result.message


def test_portal_version_parses_numeric_dev_suffix():
    from mobilerun.diagnostics.versions import parse_portal_version

    assert parse_portal_version("v0.7.26-dev") == (0, 7, 26)
