"""Device discovery, CLI target selection, and saved configuration guards."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.base import menu
from mobilerun.cli.devices import device
from mobilerun.cli.devices import list as listing
from mobilerun.cli.devices.base import discovery, options, selection
from mobilerun.cli.devices.base.discovery import DiscoveredDevice
from mobilerun.clients.local import login_store
from mobilerun.config.loader import ConfigLoader, OutdatedConfigError
from mobilerun.config.user_config import MobileConfig


@click.command()
@options.device_options
def echo_target(target, **kwargs):
    click.echo(target)


def test_device_options_expose_positional_target_and_type_flags():
    result = CliRunner().invoke(echo_target, ["--help"])
    assert result.exit_code == 0
    assert "--cloud" in result.output
    assert "DEVICE" in result.output
    assert "-a, --android" in result.output
    assert "-i, --ios" in result.output
    assert "-v, --remote" in result.output
    assert "-c, --cloud" in result.output


def test_device_type_flags_force_detected_kind_and_reject_multiple():
    from mobilerun.cli.devices.base.options import selected_device_kind

    assert (
        selected_device_kind("http://phone", False, True, False, False, None, None)
        == "ios"
    )
    assert (
        selected_device_kind("not-a-uuid", False, False, True, False, None, None)
        == "cloud"
    )
    assert (
        selected_device_kind("serial", True, False, False, False, None, None)
        == "android"
    )
    assert (
        selected_device_kind("http://remote", False, False, False, True, None, None)
        == "visual_remote"
    )
    with pytest.raises(click.UsageError, match="at most one"):
        selected_device_kind("serial", True, True, False, False, None, None)


def test_device_target_precedence(monkeypatch, tmp_path):
    config = SimpleNamespace(device=SimpleNamespace(serial="saved"))
    monkeypatch.setattr(options.ConfigLoader, "load", lambda path: config)

    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(
            return_value=[
                DiscoveredDevice("discovered", "local", None, "device", None),
                DiscoveredDevice("saved", "local", None, "device", None),
            ]
        ),
    )
    monkeypatch.setenv("MOBILERUN_DEFAULT_DEVICE", "environment")

    def command(target, **kwargs):
        click.echo(target)

    command.__module__ = "mobilerun.cli.devices.test"
    command = click.command()(options.device_options(command))
    runner = CliRunner()
    explicit_config = tmp_path / "config.yaml"
    explicit_config.touch()
    assert (
        "explicit"
        in runner.invoke(command, ["explicit", "--config", str(explicit_config)]).output
    )
    assert "saved" in runner.invoke(command, ["--config", str(explicit_config)]).output
    assert "saved" in runner.invoke(command, []).output
    monkeypatch.delenv("MOBILERUN_DEFAULT_DEVICE")
    assert "saved" in runner.invoke(command, []).output
    config.device.serial = None
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(
            return_value=[DiscoveredDevice("discovered", "local", None, "device", None)]
        ),
    )
    assert "discovered" in runner.invoke(command, []).output


def test_discovery_local_and_cloud(monkeypatch):
    async def local():
        return [SimpleNamespace(serial="usb-1", state="device")]

    async def cloud(**kwargs):
        return [
            SimpleNamespace(id="cloud-1", name="Phone", state="stopped", type="android")
        ]

    monkeypatch.setattr(discovery.adb, "list", local)
    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: "key")
    monkeypatch.setattr(discovery, "list_devices", cloud)
    devices = asyncio.run(discovery.discover_devices())
    assert [(d.id, d.source, d.ready) for d in devices] == [
        ("usb-1", "local", True),
        ("cloud-1", "cloud", False),
    ]


@pytest.mark.parametrize("interactive", [True, False])
@pytest.mark.parametrize("count", [0, 1, 2])
def test_selection_count(monkeypatch, interactive, count):
    devices = [
        DiscoveredDevice(str(i), "local", None, "device", None) for i in range(count)
    ]

    async def discover(**kwargs):
        return devices

    monkeypatch.setattr(options, "discover_devices", discover)
    monkeypatch.setattr(options, "is_interactive", lambda: interactive)
    monkeypatch.setattr(options, "select_prompt", lambda *args: "1")
    monkeypatch.setattr(options, "confirm_prompt", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial=None)),
    )

    result = CliRunner().invoke(echo_target, [])
    if count == 0:
        assert result.exit_code == 1
        assert "No devices found." in result.output
    elif count == 1:
        assert result.exit_code == 0
        assert result.stdout == "0\n"
        assert "Using 0" in result.stderr
    elif interactive:
        assert result.exit_code == 0
        assert result.stdout == "1\n"
    else:
        assert result.exit_code == 1
        assert "0 (local)" in result.output and "1 (local)" in result.output


def test_missing_key_skips_cloud(monkeypatch):
    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: None)
    assert asyncio.run(discovery.discover_devices(local=False)) == []


def test_cloud_error_warns_and_returns_local(monkeypatch, capsys):
    async def local():
        return [SimpleNamespace(serial="usb", state="device")]

    async def cloud(**kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(discovery.adb, "list", local)
    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: "key")
    monkeypatch.setattr(discovery, "list_devices", cloud)
    assert [d.id for d in asyncio.run(discovery.discover_devices())] == ["usb"]
    output = capsys.readouterr().err
    assert "--debug" in output and "offline" not in output


@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_and_default_skip_discovery(monkeypatch, explicit):
    async def fail(**kwargs):
        if explicit:
            raise AssertionError("discovery called")

        return [DiscoveredDevice("default", "local", None, "device", None)]

    monkeypatch.setattr(options, "discover_devices", fail)
    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial="default")),
    )

    result = CliRunner().invoke(echo_target, ["chosen"] if explicit else [])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == ("chosen" if explicit else "default")


def test_picker_labels_not_ready(monkeypatch):
    async def discover(**kwargs):
        return [
            DiscoveredDevice("usb", "local", None, "device", None),
            DiscoveredDevice("cloud", "cloud", "Phone", "stopped", "android"),
        ]

    labels = []

    def select(message, choices):
        labels.extend(choice.label for choice in choices)
        return "cloud"

    monkeypatch.setattr(options, "discover_devices", discover)
    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(options, "select_prompt", select)
    monkeypatch.setattr(options, "confirm_prompt", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial=None)),
    )

    result = CliRunner().invoke(echo_target)
    assert result.exit_code == 0, result.output
    assert "cloud (Phone) [stopped] cloud (not ready)" in labels


def test_cancel_and_empty_picker(monkeypatch):
    async def discover(**kwargs):
        return [
            DiscoveredDevice("a", "local", None, "device", None),
            DiscoveredDevice("b", "local", None, "device", None),
        ]

    monkeypatch.setattr(options, "discover_devices", discover)
    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial=None)),
    )

    monkeypatch.setattr(options, "select_prompt", lambda *args: None)
    assert CliRunner().invoke(echo_target).exit_code == 0

    def cancel(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(options, "select_prompt", cancel)
    assert CliRunner().invoke(echo_target).exit_code == 130


def test_device_option_wins_over_default(monkeypatch):
    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial=None)),
    )

    def command(target, **kwargs):
        click.echo(target)

    command.__module__ = "mobilerun.cli.devices.test"
    wrapped = click.command()(options.device_options(command))
    result = CliRunner().invoke(wrapped, ["--device", "chosen"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "chosen\n"


def test_cloud_listing_uses_settings_base_url(monkeypatch):
    from mobilerun.clients.cloud import client

    monkeypatch.setenv("MOBILERUN_CLOUD_BASE_URL", "https://configured.example")
    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: "fake-key")
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.devices = SimpleNamespace(list=self._list)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def _list(self, **kwargs):
            return SimpleNamespace(items=[], pagination=SimpleNamespace(has_next=False))

    monkeypatch.setattr(client, "AsyncMobilerun", FakeClient)
    asyncio.run(discovery.discover_devices(local=False))
    assert calls[0]["base_url"] == "https://configured.example"


def test_list_human_and_json(monkeypatch):
    async def discover(**kwargs):
        return [
            DiscoveredDevice("usb", "local", None, "device", None),
            DiscoveredDevice("cloud", "cloud", "Phone", "ready", "android"),
        ]

    monkeypatch.setattr(listing, "discover_devices", discover)
    runner = CliRunner()
    result = runner.invoke(listing.devices, ["--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == [
        {"id": "usb", "source": "local", "default": False, "state": "device"},
        {
            "id": "cloud",
            "source": "cloud",
            "default": False,
            "name": "Phone",
            "state": "ready",
            "type": "android",
        },
    ]
    result = runner.invoke(listing.devices, [])
    assert result.exit_code == 0, result.output
    assert "Local devices · 1" in result.stdout
    assert "usb" in result.stdout
    assert "Cloud devices · 1" in result.stdout
    assert "Phone    cloud    ● ready    android" in result.stdout


def test_cloud_list_errors_are_fatal(monkeypatch):
    async def fail(**kwargs):
        raise RuntimeError("private upstream detail")

    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: "key")
    monkeypatch.setattr(discovery, "list_devices", fail)
    result = CliRunner().invoke(listing.devices, ["--cloud"])
    assert result.exit_code == 1
    assert "Error listing cloud devices: offline" in result.stderr
    assert "private upstream detail" not in result.stderr


def test_cloud_list_uses_public_discovery(monkeypatch):
    async def discover(*, local, cloud, base_url, billing, strict):
        assert local is False and cloud is True and strict is True
        return [DiscoveredDevice("cloud-1", "cloud", "Phone", "ready", "android")]

    monkeypatch.setattr(listing, "discover_devices", discover)
    result = CliRunner().invoke(listing.devices, ["--cloud", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)[0]["id"] == "cloud-1"


def test_device_picker_precedes_coordinates(monkeypatch):
    order = []
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(
            return_value=[
                DiscoveredDevice("a", "local", None, "device", None),
                DiscoveredDevice("b", "local", None, "device", None),
            ]
        ),
    )
    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    def pick(*args):
        order.append("device")
        return "a"

    def text(label, **kwargs):
        order.append(label)
        return "1"

    monkeypatch.setattr(options, "select_prompt", pick)
    monkeypatch.setattr(menu, "text_prompt", text)
    monkeypatch.setattr(options, "confirm_prompt", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        selection, "connect_target", AsyncMock(return_value=AsyncMock())
    )
    result = CliRunner().invoke(device, ["input", "tap"])
    assert result.exit_code == 0, result.output
    assert order == ["device", "X coordinate", "Y coordinate"]


def test_list_marks_default(monkeypatch):
    monkeypatch.setattr(
        listing,
        "discover_devices",
        AsyncMock(
            return_value=[
                DiscoveredDevice("a", "local", None, "device", None),
                DiscoveredDevice("b", "cloud", "Phone", "ready", "android"),
            ]
        ),
    )
    monkeypatch.setattr(
        listing.ConfigLoader,
        "load",
        lambda *args: SimpleNamespace(device=SimpleNamespace(serial="b")),
    )
    runner = CliRunner()
    result = runner.invoke(listing.devices, ["--json"])
    assert [item["default"] for item in json.loads(result.stdout)] == [False, True]
    result = runner.invoke(listing.devices)
    assert "★" in result.stdout and "b" in result.stdout


def test_unavailable_saved_default_errors_without_fallback(monkeypatch):
    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial="missing")),
    )
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(
            return_value=[DiscoveredDevice("other", "local", None, "device", None)]
        ),
    )
    monkeypatch.setattr(options, "is_interactive", lambda: False)

    result = CliRunner().invoke(echo_target)
    assert result.exit_code == 1
    assert "Default device missing is not available." in result.output


def test_picker_save_yes_and_no(monkeypatch):
    devices = [
        DiscoveredDevice("a", "local", None, "device", None),
        DiscoveredDevice("b", "local", None, "device", None),
    ]
    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial=None)),
    )
    monkeypatch.setattr(options, "discover_devices", AsyncMock(return_value=devices))
    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(options, "select_prompt", lambda *args: "a")
    saved = []
    monkeypatch.setattr(options, "save_default_device", saved.append)

    for answer in (False, True):
        monkeypatch.setattr(
            options, "confirm_prompt", lambda *args, answer=answer, **kwargs: answer
        )
        result = CliRunner().invoke(echo_target)
        assert result.exit_code == 0, result.output
    assert saved == ["a"]


def test_existing_default_no_save_prompt(monkeypatch):
    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial="a")),
    )
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(return_value=[DiscoveredDevice("a", "local", None, "device", None)]),
    )
    monkeypatch.setattr(options, "is_interactive", lambda: True)

    def fail(*args, **kwargs):
        raise AssertionError("unexpected confirmation")

    monkeypatch.setattr(options, "confirm_prompt", fail)

    result = CliRunner().invoke(echo_target)
    assert result.exit_code == 0, result.output
    assert result.stdout == "a\n"


def test_explicit_device_option_precedes_prompt_without_discovery(monkeypatch):
    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(side_effect=AssertionError("unexpected discovery")),
    )
    monkeypatch.setattr(menu, "text_prompt", lambda *args: "1")
    driver = AsyncMock()
    connect = AsyncMock(return_value=driver)
    monkeypatch.setattr(selection, "connect_target", connect)
    result = CliRunner().invoke(device, ["input", "tap", "--device", "chosen"])
    assert result.exit_code == 0, result.output
    assert connect.await_args.args[0] == "chosen"


@pytest.fixture
def clean_target(monkeypatch):
    monkeypatch.delenv("ANDROID_SERIAL", raising=False)
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)


@pytest.fixture
def adb_phone(monkeypatch):
    def set_devices(**states):
        monkeypatch.setattr(
            discovery.adb,
            "list",
            AsyncMock(
                return_value=[
                    SimpleNamespace(serial=k, state=v) for k, v in states.items()
                ]
            ),
        )

    return set_devices


@pytest.mark.usefixtures("clean_target")
@pytest.mark.parametrize(
    "serial,explicit,expected",
    [
        ("missing", None, None),
        ("wanted", None, "wanted"),
        ("missing", "chosen", "chosen"),
    ],
)
def test_file_rm_respects_target_precedence(
    monkeypatch, adb_phone, serial, explicit, expected
):
    cfg = MobileConfig()
    cfg.device.serial = "saved"
    ConfigLoader.save(cfg)
    monkeypatch.setenv("ANDROID_SERIAL", serial)
    adb_phone(saved="device", wanted="device")
    connect = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(selection, "connect_target", connect)
    args = ["file", "rm", "/demo", "--yes"]
    result = CliRunner().invoke(
        device, args + (["--device", explicit] if explicit else [])
    )
    if expected is None:
        connect.assert_not_awaited()
        assert result.exit_code == 1 and serial in result.stderr
    else:
        assert result.exit_code == 0, result.output
        assert connect.await_args.args == (expected,)


@pytest.mark.usefixtures("clean_target")
@pytest.mark.parametrize("online", [True, False])
def test_only_online_adb_device_is_selected(monkeypatch, adb_phone, online):
    adb_phone(
        off="offline", locked="unauthorized", **({"on": "device"} if online else {})
    )
    connect = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(selection, "connect_target", connect)
    result = CliRunner().invoke(device, ["file", "rm", "/demo", "--yes"])
    if online:
        assert result.exit_code == 0, result.output
        assert connect.await_args.args == ("on",)
    else:
        assert result.exit_code == 1 and "No devices found" in result.stderr
        connect.assert_not_awaited()


@pytest.mark.usefixtures("clean_target")
@pytest.mark.parametrize("args", [[], ["--json"], ["--cloud"]])
def test_bad_login_does_not_hide_local_state(adb_phone, tmp_path, monkeypatch, args):
    adb_phone(locked="unauthorized")
    path = tmp_path / "bad.json"
    path.write_text("{")
    monkeypatch.setattr(login_store, "credential_path", lambda: path)
    result = CliRunner().invoke(device, ["list", *args])
    assert "credentials" in result.stderr.lower()
    if "--cloud" in args:
        assert result.exit_code == 1
    elif "--json" in args:
        assert result.exit_code == 0
        assert json.loads(result.stdout)[0]["state"] == "unauthorized"
    else:
        assert result.exit_code == 0 and "unauthorized" in result.stdout


@pytest.mark.usefixtures("clean_target")
def test_ios_does_not_receive_saved_android_token(monkeypatch):
    cfg = MobileConfig()
    cfg.device.auth_token = "android-test-token"
    create = Mock(return_value=AsyncMock())
    monkeypatch.setattr(selection, "IOSDevice", create)
    asyncio.run(selection.connect_target("http://ios", kind="ios", config=cfg))
    create.assert_called_once_with("http://ios", token=None)


@pytest.mark.usefixtures("clean_target")
def test_android_ping_does_not_receive_saved_ios_token(monkeypatch):
    from importlib import import_module

    ping = import_module("mobilerun.cli.devices.ping")
    cfg = MobileConfig()
    cfg.device.platform = "ios"
    cfg.device.auth_token = "ios-test-token"
    ConfigLoader.save(cfg)
    create = Mock(return_value=AsyncMock())
    monkeypatch.setattr(ping, "create_android_portal_http_driver", create)
    result = CliRunner().invoke(device, ["ping", "--android", "http://portal"])
    assert result.exit_code == 0, result.output
    create.assert_called_once_with("http://portal", None)


@pytest.mark.usefixtures("clean_target")
def test_missing_env_config_never_uses_global(tmp_path, monkeypatch):
    ConfigLoader.save(MobileConfig())
    monkeypatch.setenv("MOBILERUN_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(OutdatedConfigError, match="missing.yaml"):
        ConfigLoader.load()


@pytest.mark.usefixtures("clean_target")
@pytest.mark.parametrize("value", ["", " \t "])
def test_blank_env_config_uses_saved_user_config(monkeypatch, value):
    config = MobileConfig()
    config.device.serial = "saved"
    ConfigLoader.save(config)
    monkeypatch.setenv("MOBILERUN_CONFIG", value)
    assert ConfigLoader.load().device.serial == "saved"


@pytest.mark.usefixtures("clean_target")
def test_saved_token_and_env_override(monkeypatch):
    cfg = MobileConfig().device
    cfg.platform = "ios"
    cfg.auth_token = "ios-test-token"
    assert cfg.resolve_auth_token("android") is None
    assert cfg.resolve_auth_token("ios") == "ios-test-token"
    monkeypatch.setenv("MOBILERUN_DEVICE_TOKEN", "test-override")
    assert cfg.resolve_auth_token("android") == "test-override"
