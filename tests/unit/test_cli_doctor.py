from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner


@pytest.mark.parametrize("flag", ["--device", "-d"])
def test_doctor_legacy_device_option(monkeypatch, flag):
    import importlib

    module = importlib.import_module("mobilerun.cli.doctor")
    run = AsyncMock()
    monkeypatch.setattr(module, "run_doctor", run)
    result = CliRunner().invoke(module.doctor, [flag, "emulator-5554"])
    assert result.exit_code == 0, result.output
    run.assert_awaited_once_with("emulator-5554", False, fix=False)


def test_doctor_passes_flags_to_diagnostics(monkeypatch):
    import importlib

    doctor_module = importlib.import_module("mobilerun.cli.doctor")
    from mobilerun.cli.doctor import doctor

    run = AsyncMock()
    monkeypatch.setattr(doctor_module, "run_doctor", run)
    result = CliRunner().invoke(doctor, ["emulator-5554", "--android", "--debug"])
    assert result.exit_code == 0, result.output
    run.assert_awaited_once_with("emulator-5554", True, fix=False)


def test_doctor_portal_http_android_runs_local_only(monkeypatch):
    import importlib

    doctor_module = importlib.import_module("mobilerun.cli.doctor")
    from mobilerun.cli.doctor import doctor

    run = AsyncMock()
    monkeypatch.setattr(doctor_module, "run_doctor", run)
    result = CliRunner().invoke(doctor, ["http://127.0.0.1:8080", "-a"])
    assert result.exit_code == 0, result.output
    assert " ".join(result.output.split()) == (
        "! doctor: device checks for Portal HTTP targets aren't supported yet; ran local checks"
    )
    run.assert_awaited_once_with(None, False, fix=False)


def test_doctor_without_target_uses_discovery(monkeypatch):
    import importlib

    doctor_module = importlib.import_module("mobilerun.cli.doctor")
    from mobilerun.cli.doctor import doctor

    run = AsyncMock()
    monkeypatch.setattr(doctor_module, "run_doctor", run)
    from mobilerun.clients.portal import discovery

    monkeypatch.setattr(
        discovery.adb,
        "list",
        AsyncMock(
            return_value=[
                __import__("types").SimpleNamespace(
                    serial="test-device", state="device"
                )
            ]
        ),
    )
    result = CliRunner().invoke(doctor, [])
    assert result.exit_code == 0, result.output
    assert "Using test-device" in result.stderr
    run.assert_awaited_once_with("test-device", False, fix=False)


def test_doctor_rejects_multiple_targets():
    from mobilerun.cli.doctor import doctor

    result = CliRunner().invoke(doctor, ["serial", "--android", "--ios"])
    assert result.exit_code == 2
    assert "Choose at most one device type flag" in result.output


@pytest.mark.parametrize("kind", ["ios", "cloud", "remote"])
def test_non_android_doctor_runs_local_checks_only(monkeypatch, kind):
    import importlib

    doctor_module = importlib.import_module("mobilerun.cli.doctor")
    from mobilerun.cli.doctor import doctor

    run = AsyncMock()
    monkeypatch.setattr(doctor_module, "run_doctor", run)
    result = CliRunner().invoke(doctor, ["target", f"--{kind}"])
    assert result.exit_code == 0, result.output
    assert (
        "doctor: device checks support Android only; ran local checks" in result.output
    )
    run.assert_awaited_once_with(None, False, fix=False)


def test_tcp_check_reuses_existing_forward(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from mobilerun.diagnostics import checks

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

    monkeypatch.setattr(checks.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(
        checks.PortalClient, "_fetch_auth_token", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(checks.httpx, "AsyncClient", Client)
    result = asyncio.run(checks.check_tcp(Device(), False))
    assert result.status == checks.Status.PASS, result.message
