from unittest.mock import AsyncMock

from click.testing import CliRunner

from mobilerun.cli.devices.base.discovery import DiscoveredDevice


def test_connect_discovered_sets_default(monkeypatch):
    from mobilerun.cli.devices import connect, device

    monkeypatch.setattr(
        connect,
        "discover_devices",
        AsyncMock(return_value=[DiscoveredDevice("usb", "local", None, None, None)]),
    )
    saved = []
    monkeypatch.setattr(connect, "save_default_device", saved.append)
    adb_connect = AsyncMock()
    monkeypatch.setattr(connect.adb, "connect", adb_connect)
    result = CliRunner().invoke(device, ["connect", "usb"])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "✓ Connected usb (default device)"
    assert saved == ["usb"]
    adb_connect.assert_not_awaited()


def test_connect_picker_add_address(monkeypatch):
    from mobilerun.cli.devices import connect, device

    monkeypatch.setattr(connect, "is_interactive", lambda: True)
    monkeypatch.setattr(connect, "discover_devices", AsyncMock(return_value=[]))
    labels = []

    async def select(message, choices):
        labels.extend(c.label for c in choices)
        return "__add_address__"

    monkeypatch.setattr(connect, "select_prompt", select)
    monkeypatch.setattr(connect, "text_prompt", AsyncMock(return_value="192.168.1.2"))
    monkeypatch.setattr(
        connect.adb, "connect", AsyncMock(return_value="connected to 192.168.1.2:5555")
    )
    saved = []
    monkeypatch.setattr(connect, "save_default_device", saved.append)
    result = CliRunner().invoke(device, ["connect"])
    assert result.exit_code == 0, result.output
    assert labels[-1] == "Add device by address (host:port)…"
    assert saved == ["192.168.1.2:5555"]
    assert "✓ Connected 192.168.1.2:5555 (default device)" in result.stdout


def test_connect_without_target_noninteractive(monkeypatch):
    from mobilerun.cli.devices import connect, device

    monkeypatch.setattr(connect, "is_interactive", lambda: False)
    result = CliRunner().invoke(device, ["connect"])
    assert result.exit_code == 1
    assert "Pass a device id or host:port." in result.output


def test_disconnect_default_tcp(monkeypatch):
    from types import SimpleNamespace

    from mobilerun.cli.devices import device, disconnect

    monkeypatch.setattr(
        disconnect.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial="1.2.3.4:5555")),
    )
    adb_disconnect = AsyncMock(return_value="disconnected 1.2.3.4:5555")
    monkeypatch.setattr(disconnect.adb, "disconnect", adb_disconnect)
    saved = []
    monkeypatch.setattr(disconnect, "save_default_device", saved.append)
    result = CliRunner().invoke(device, ["disconnect"])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "✓ Disconnected 1.2.3.4:5555"
    assert saved == [None]
    adb_disconnect.assert_awaited_once_with("1.2.3.4:5555", raise_error=True)


def test_connect_disconnect_preserve_config(monkeypatch, tmp_path):
    import yaml

    from mobilerun.cli.devices import connect, device
    from mobilerun.config.migrations import CURRENT_VERSION

    path = tmp_path / "config.yaml"
    path.write_text(
        f"_version: {CURRENT_VERSION}\ndevice:\n  serial: old\n  use_tcp: true\ncustom: retained\n"
    )
    monkeypatch.setenv("MOBILERUN_CONFIG", str(path))
    monkeypatch.setattr(
        connect,
        "discover_devices",
        AsyncMock(return_value=[DiscoveredDevice("usb", "local", None, None, None)]),
    )
    runner = CliRunner()
    result = runner.invoke(device, ["connect", "usb"])
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(path.read_text())
    assert data["device"] == {"serial": "usb", "use_tcp": True}
    assert data["custom"] == "retained"
    result = runner.invoke(device, ["disconnect"])
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(path.read_text())
    assert data["device"].get("serial") is None and data["device"]["use_tcp"]
    assert data["custom"] == "retained"


def test_connect_unreachable_does_not_save(monkeypatch):
    from mobilerun.cli.devices import connect, device

    monkeypatch.setattr(connect, "discover_devices", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        connect.adb, "connect", AsyncMock(return_value="failed to connect")
    )
    saved = []
    monkeypatch.setattr(connect, "save_default_device", saved.append)
    result = CliRunner().invoke(device, ["connect", "host:5555"])
    assert result.exit_code == 1
    assert "Failed to connect to host:5555" in result.output
    assert saved == []


def test_disconnect_specific_other_tcp_keeps_default(monkeypatch):
    from types import SimpleNamespace

    from mobilerun.cli.devices import device, disconnect

    monkeypatch.setattr(
        disconnect.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial="usb")),
    )
    monkeypatch.setattr(
        disconnect.adb, "disconnect", AsyncMock(return_value="disconnected host:5555")
    )
    saved = []
    monkeypatch.setattr(disconnect, "save_default_device", saved.append)
    result = CliRunner().invoke(device, ["disconnect", "host:5555"])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "✓ Disconnected host:5555"
    assert saved == []


def test_disconnect_no_default(monkeypatch):
    from types import SimpleNamespace

    from mobilerun.cli.devices import device, disconnect

    monkeypatch.setattr(
        disconnect.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial=None)),
    )
    result = CliRunner().invoke(device, ["disconnect"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "No default device."


def test_connect_picker_selects_discovered_device(monkeypatch):
    from mobilerun.cli.devices import connect, device

    monkeypatch.setattr(connect, "is_interactive", lambda: True)
    monkeypatch.setattr(
        connect,
        "discover_devices",
        AsyncMock(return_value=[DiscoveredDevice("usb", "local", None, None, None)]),
    )
    monkeypatch.setattr(connect, "select_prompt", AsyncMock(return_value="usb"))
    monkeypatch.setattr(
        connect.adb,
        "connect",
        AsyncMock(side_effect=AssertionError("unexpected TCP connection")),
    )
    saved = []
    monkeypatch.setattr(connect, "save_default_device", saved.append)
    result = CliRunner().invoke(device, ["connect"])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "✓ Connected usb (default device)"
    assert saved == ["usb"]


def test_disconnect_failure_does_not_clear_default(monkeypatch):
    from types import SimpleNamespace

    from mobilerun.cli.devices import device, disconnect

    monkeypatch.setattr(
        disconnect.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial="host:5555")),
    )
    monkeypatch.setattr(
        disconnect.adb, "disconnect", AsyncMock(side_effect=OSError("offline"))
    )
    saved = []
    monkeypatch.setattr(disconnect, "save_default_device", saved.append)
    result = CliRunner().invoke(device, ["disconnect"])
    assert result.exit_code == 1
    assert "--debug" in result.output and "offline" not in result.output
    assert saved == []
