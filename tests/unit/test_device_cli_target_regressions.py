"""Device command parsing, config persistence, and migration examples."""

import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from click.testing import CliRunner
from mobilerun_core_local.driver import ios

from mobilerun.cli.devices import device
from mobilerun.cli.devices.base import options, selection
from mobilerun.cli.devices.base.discovery import DiscoveredDevice
from mobilerun.config.migrations import CURRENT_VERSION
from mobilerun.config.user_config import MobileConfig

CLOUD = "11111111-1111-1111-1111-111111111111"


@pytest.mark.parametrize(
    "kind,expected", [("ios", "http://ios.test"), ("android", "usb"), ("cloud", CLOUD)]
)
def test_interactive_discovery_honors_kind(monkeypatch, kind, expected):
    async def discover(*, local=True, cloud=True, **kwargs):
        return (
            [DiscoveredDevice("usb", "local", None, None, None)] if local else []
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
        DiscoveredDevice(id, "local", None, None, None) for id in ("usb-a", "usb-b")
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
    readme = (
        Path(__file__).resolve().parents[2] / "mobilerun/cli/devices/README.md"
    ).read_text()
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
