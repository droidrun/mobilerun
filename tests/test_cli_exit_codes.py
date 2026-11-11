from unittest.mock import AsyncMock

from click.testing import CliRunner

from mobilerun.cli import main


def test_connect_fresh_success(monkeypatch):
    connect = AsyncMock(return_value="connected to 127.0.0.1:5555")
    monkeypatch.setattr(main.adb, "connect", connect)

    result = CliRunner().invoke(main.cli, ["connect", "127.0.0.1:5555"])

    assert result.exit_code == 0, result.output
    assert "Successfully connected to 127.0.0.1:5555" in result.output
    connect.assert_awaited_once_with("127.0.0.1:5555")


def test_connect_already_connected(monkeypatch):
    monkeypatch.setattr(
        main.adb, "connect", AsyncMock(return_value="already connected to device")
    )

    result = CliRunner().invoke(main.cli, ["connect", "device"])

    assert result.exit_code == 0, result.output
    assert "Successfully connected to device" in result.output


def test_connect_failure_string(monkeypatch):
    monkeypatch.setattr(
        main.adb, "connect", AsyncMock(return_value="connection refused")
    )

    result = CliRunner().invoke(main.cli, ["connect", "device"])

    assert result.exit_code == 1
    assert "Error: Failed to connect to device: connection refused" in result.output


def test_connect_exception(monkeypatch):
    monkeypatch.setattr(
        main.adb, "connect", AsyncMock(side_effect=RuntimeError("connection broke"))
    )

    result = CliRunner().invoke(main.cli, ["connect", "device"])

    assert result.exit_code == 1
    assert "Error: Error connecting to device: connection broke" in result.output


def test_disconnect_exception(monkeypatch):
    disconnect = AsyncMock(side_effect=RuntimeError("connection broke"))
    monkeypatch.setattr(main.adb, "disconnect", disconnect)

    result = CliRunner().invoke(main.cli, ["disconnect", "device"])

    assert result.exit_code == 1
    assert "Error: Error disconnecting from device: connection broke" in result.output
    disconnect.assert_awaited_once_with("device", raise_error=True)


def test_disconnect_not_connected_is_success(monkeypatch):
    monkeypatch.setattr(main.adb, "disconnect", AsyncMock(return_value=False))

    result = CliRunner().invoke(main.cli, ["disconnect", "device"])

    assert result.exit_code == 0, result.output
    assert "Device device was not connected" in result.output


def test_ping_unknown_device(monkeypatch):
    monkeypatch.setattr(main.adb, "device", AsyncMock(return_value=None))

    result = CliRunner().invoke(main.cli, ["ping", "-d", "no-such-device"])

    assert result.exit_code == 1
    assert "Error: Could not find device no-such-device" in result.output


def test_ping_driver_error(monkeypatch):
    monkeypatch.setattr(main.adb, "device", AsyncMock(return_value=object()))
    monkeypatch.setattr(
        main, "ping_portal", AsyncMock(side_effect=RuntimeError("portal broke"))
    )

    result = CliRunner().invoke(main.cli, ["ping", "-d", "device", "--debug"])

    assert result.exit_code == 1
    assert "Error: portal broke" in result.output
    assert "Traceback (most recent call last):" in result.output


def test_devices_local_error_still_lists_cloud(monkeypatch):
    monkeypatch.setattr(
        main.adb, "list", AsyncMock(side_effect=RuntimeError("adb broke"))
    )
    monkeypatch.setattr(main, "resolve_cloud_api_key", lambda: "test-token")
    cloud_listing = AsyncMock(return_value=0)
    monkeypatch.setattr(main, "_list_cloud_devices", cloud_listing)

    result = CliRunner().invoke(main.cli, ["devices"])

    assert result.exit_code == 1
    assert "Error listing local devices: adb broke" in result.output
    cloud_listing.assert_awaited_once_with(
        None, silent_if_no_key=True, silent_errors=True
    )


def test_run_provider_without_model(monkeypatch):
    run_command = AsyncMock()
    monkeypatch.setattr(main, "run_command", run_command)

    result = CliRunner().invoke(main.cli, ["run", "x", "--provider", "OpenAI"])

    assert result.exit_code == 2
    assert "--provider and --model must be given together" in result.output
    run_command.assert_not_awaited()


def test_run_model_without_provider(monkeypatch):
    run_command = AsyncMock()
    monkeypatch.setattr(main, "run_command", run_command)

    result = CliRunner().invoke(main.cli, ["run", "x", "--model", "example"])

    assert result.exit_code == 2
    assert "--provider and --model must be given together" in result.output
    run_command.assert_not_awaited()
