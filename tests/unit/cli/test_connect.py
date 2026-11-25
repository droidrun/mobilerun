"""Device command parsing, config persistence, and migration examples."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from click.testing import CliRunner
from mobilerun_core_local.driver import ios

from mobilerun.cli.devices import connect, device, disconnect
from mobilerun.cli.devices.base import options, selection
from mobilerun.cli.devices.base.discovery import DiscoveredDevice
from mobilerun.config.migrations import CURRENT_VERSION
from mobilerun.config.user_config import MobileConfig
from tests.support.paths import REPO_ROOT

CLOUD = "11111111-1111-1111-1111-111111111111"


def test_connect_discovered_sets_default(monkeypatch):
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
    monkeypatch.setattr(connect, "is_interactive", lambda: False)
    result = CliRunner().invoke(device, ["connect"])
    assert result.exit_code == 1
    assert "Pass a device id or host:port." in result.output


def test_disconnect_default_tcp(monkeypatch):
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
    monkeypatch.setattr(
        disconnect.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial=None)),
    )
    result = CliRunner().invoke(device, ["disconnect"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "No default device."


def test_connect_picker_selects_discovered_device(monkeypatch):
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


def test_connect_failure_exits_nonzero(monkeypatch):
    monkeypatch.setattr(connect, "discover_devices", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        "mobilerun.cli.devices.connect.adb.connect", AsyncMock(return_value="failed")
    )
    result = CliRunner().invoke(connect.connect, ["localhost:5555"])
    assert result.exit_code == 1


@pytest.mark.parametrize(
    "kind,expected", [("ios", "http://ios.test"), ("android", "usb"), ("cloud", CLOUD)]
)
def test_interactive_discovery_honors_kind(monkeypatch, kind, expected):
    async def discover(*, local=True, cloud=True, **kwargs):
        return (
            [DiscoveredDevice("usb", "local", None, "device", None)] if local else []
        ) + (
            [DiscoveredDevice(CLOUD, "cloud", None, None, None)]
            if cloud and kind != "ios"
            else []
        )

    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(
        options, "select_prompt", lambda *args: pytest.fail("unexpected picker")
    )
    monkeypatch.setattr(options, "discover_devices", discover)
    monkeypatch.setattr(
        ios, "discover_ios_device", AsyncMock(return_value="http://ios.test")
    )
    connect = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(selection, "connect_target", connect)
    result = CliRunner().invoke(device, ["input", "clipboard", "get", f"--{kind}"])
    assert result.exit_code == 0, result.output
    assert connect.await_args.args == (expected,)
    assert connect.await_args.kwargs["kind"] == kind


def test_ios_skips_incompatible_saved_android_serial(monkeypatch):
    config = MobileConfig()
    config.device.serial = "usb"
    monkeypatch.setattr(options.ConfigLoader, "load", lambda path: config)
    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(side_effect=AssertionError("unexpected ADB discovery")),
    )
    monkeypatch.setattr(
        ios, "discover_ios_device", AsyncMock(return_value="http://ios.test")
    )
    connect = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(selection, "connect_target", connect)
    result = CliRunner().invoke(device, ["input", "tap", "10", "20", "--ios"])
    assert result.exit_code == 0, result.output
    assert connect.await_args.args == ("http://ios.test",)


def test_remote_does_not_fall_back_to_adb(monkeypatch):
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(side_effect=AssertionError("unexpected discovery")),
    )
    result = CliRunner().invoke(device, ["input", "clipboard", "get", "--remote"])
    assert result.exit_code == 2
    assert "remote device URL" in result.output


@pytest.mark.parametrize("device_args", [[CLOUD], ["-d", CLOUD], ["--device", CLOUD]])
def test_rename_positional_or_device_option(monkeypatch, device_args):
    module = importlib.import_module("mobilerun.cli.devices.cloud.rename")
    rename = AsyncMock()
    monkeypatch.setattr(module.devices, "set_name", rename)
    monkeypatch.setattr(module, "cloud_client", MagicMock())
    result = CliRunner().invoke(device, ["cloud", "rename", *device_args, "New"])
    assert result.exit_code == 0, result.output
    assert rename.await_args.args == (CLOUD, "New")


def test_save_to_explicit_config(monkeypatch, tmp_path):
    alternate = tmp_path / "alternate.yaml"
    alternate.write_text(f"_version: {CURRENT_VERSION}\ndevice: {{}}\n")
    default = options.ConfigLoader.get_user_config_path()
    assert default.is_relative_to(tmp_path)
    options.ConfigLoader.load(None)
    before = default.read_bytes()
    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(options, "select_prompt", lambda *args: "usb-b")
    monkeypatch.setattr(options, "confirm_prompt", lambda *args, **kwargs: True)
    devices = [
        DiscoveredDevice(id, "local", None, "device", None) for id in ("usb-a", "usb-b")
    ]
    monkeypatch.setattr(options, "discover_devices", AsyncMock(return_value=devices))
    monkeypatch.setattr(
        selection, "connect_target", AsyncMock(return_value=AsyncMock())
    )
    result = CliRunner().invoke(
        device, ["input", "clipboard", "get", "--config", str(alternate)]
    )
    assert result.exit_code == 0, result.output
    assert default.read_bytes() == before
    assert yaml.safe_load(alternate.read_text())["device"]["serial"] == "usb-b"


def test_documented_flat_command_replacements():
    readme = (REPO_ROOT / "mobilerun/cli/devices/README.md").read_text()
    for old, group, new in [
        ("screenshot", "screen", "screenshot"),
        ("ui", "screen", "ui"),
        *(
            (cmd, "input", cmd)
            for cmd in ("tap", "swipe", "long-press", "type", "press")
        ),
        ("apps", "app", "list"),
        ("start", "app", "start"),
    ]:
        assert CliRunner().invoke(device, [old, "--help"]).exit_code == 2
        assert CliRunner().invoke(device, [group, new, "--help"]).exit_code == 0
        assert f"`device {old}` | `device {group} {new}`" in readme
