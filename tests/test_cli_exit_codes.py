import importlib
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from mobilerun.cli import device_lifecycle, main, selection
from mobilerun.config_manager import MobileConfig

run_module = importlib.import_module("mobilerun.cli.run")


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setattr("mobilerun.cli.options.require_adb", lambda: "/fake/adb")
    monkeypatch.setattr(selection.ConfigLoader, "load", lambda _: MobileConfig())


def test_connect_fresh_success(monkeypatch):
    connect = AsyncMock(return_value="connected to 127.0.0.1:5555")
    monkeypatch.setattr(device_lifecycle.adb, "connect", connect)

    result = CliRunner().invoke(main.cli, ["device", "connect", "127.0.0.1:5555"])

    assert result.exit_code == 0, result.output
    assert "Successfully connected to 127.0.0.1:5555" in result.output
    connect.assert_awaited_once_with("127.0.0.1:5555")


def test_connect_already_connected(monkeypatch):
    monkeypatch.setattr(
        device_lifecycle.adb, "connect", AsyncMock(return_value="already connected to device")
    )

    result = CliRunner().invoke(main.cli, ["device", "connect", "device"])

    assert result.exit_code == 0, result.output
    assert "Successfully connected to device" in result.output


def test_connect_failure_string(monkeypatch):
    monkeypatch.setattr(
        device_lifecycle.adb, "connect", AsyncMock(return_value="connection refused")
    )

    result = CliRunner().invoke(main.cli, ["device", "connect", "device"])

    assert result.exit_code == 1
    assert "Error: Failed to connect to device: connection refused" in result.output


def test_connect_exception(monkeypatch):
    monkeypatch.setattr(
        device_lifecycle.adb, "connect", AsyncMock(side_effect=RuntimeError("connection broke"))
    )

    result = CliRunner().invoke(main.cli, ["device", "connect", "device"])

    assert result.exit_code == 1
    assert "Error: connection broke" in result.output


def test_disconnect_exception(monkeypatch):
    disconnect = AsyncMock(side_effect=RuntimeError("connection broke"))
    monkeypatch.setattr(device_lifecycle.adb, "disconnect", disconnect)

    result = CliRunner().invoke(main.cli, ["device", "disconnect", "device"])

    assert result.exit_code == 1
    assert "Error: connection broke" in result.output
    disconnect.assert_awaited_once_with("device", raise_error=True)


def test_disconnect_not_connected_is_success(monkeypatch):
    monkeypatch.setattr(device_lifecycle.adb, "disconnect", AsyncMock(return_value=False))

    result = CliRunner().invoke(main.cli, ["device", "disconnect", "device"])

    assert result.exit_code == 0, result.output
    assert "Device device was not connected" in result.output


def test_ping_unknown_device(monkeypatch):
    monkeypatch.setattr(device_lifecycle, "probe_device", lambda *_: (_ for _ in ()).throw(RuntimeError("unknown device")))
    result = CliRunner().invoke(main.cli, ["device", "ping", "-d", "no-such-device"])
    assert result.exit_code == 1
    assert "Error: unknown device" in result.output


def test_ping_driver_error(monkeypatch):
    monkeypatch.setattr(device_lifecycle, "probe_device", lambda *_: (_ for _ in ()).throw(RuntimeError("portal broke")))
    result = CliRunner().invoke(main.cli, ["device", "ping", "-d", "device", "--verbose"])
    assert result.exit_code == 1
    assert "Error: portal broke" in result.output
    assert "Traceback" in result.stderr


def test_device_list_core_error_is_failure(monkeypatch):
    class FakeCore:
        def list_devices(self, **kwargs):
            raise RuntimeError("adb broke")

    monkeypatch.setattr(device_lifecycle, "core_client", lambda *_, **kw: FakeCore())
    result = CliRunner().invoke(main.cli, ["device", "list"])
    assert result.exit_code == 1
    assert "adb broke" in result.output


def test_run_provider_without_model(monkeypatch):
    run_command = AsyncMock()
    monkeypatch.setattr(run_module, "run_command", run_command)

    result = CliRunner().invoke(main.cli, ["run", "x", "--provider", "OpenAI"])

    assert result.exit_code == 2
    assert "--provider and --model must be given together" in result.output
    run_command.assert_not_awaited()


def test_run_model_without_provider(monkeypatch):
    run_command = AsyncMock()
    monkeypatch.setattr(run_module, "run_command", run_command)

    result = CliRunner().invoke(main.cli, ["run", "x", "--model", "example"])

    assert result.exit_code == 2
    assert "--provider and --model must be given together" in result.output
    run_command.assert_not_awaited()
