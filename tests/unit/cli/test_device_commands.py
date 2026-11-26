"""Grouped CLI contract through real Click paths, with no device/cloud access."""

import asyncio
import importlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.cli.devices.base import selection
from mobilerun.cli.devices.group import device
from mobilerun.models.billing import Billing

CLOUD_ID = "11111111-1111-1111-1111-111111111111"


@pytest.mark.parametrize(
    "path,method,args,output",
    [
        (
            ["app", "stop", "org.example"],
            "stop_app",
            ("org.example",),
            "Stopped org.example",
        ),
        (
            ["app", "open", "https://example.com"],
            "open_url",
            ("https://example.com",),
            "Opened https://example.com",
        ),
        (
            ["settings", "location", "set", "52.52", "13.405"],
            "set_location",
            (52.52, 13.405),
            "Location set to 52.52, 13.405 on serial",
        ),
        (
            ["settings", "location", "reset"],
            "reset_location",
            (),
            "Location reset on serial",
        ),
        (["settings", "language", "get"], "get_language", (), "de-DE"),
        (
            ["settings", "language", "set", "de-DE"],
            "set_language",
            ("de-DE",),
            "Language set to de-DE on serial",
        ),
        (["settings", "timezone", "get"], "get_timezone", (), "Europe/Berlin"),
        (
            ["settings", "timezone", "set", "Europe/Berlin"],
            "set_timezone",
            ("Europe/Berlin",),
            "Timezone set to Europe/Berlin on serial",
        ),
        (
            ["input", "clipboard", "set", "private text"],
            "set_clipboard",
            ("private text",),
            f"Clipboard set on {CLOUD_ID}",
        ),
    ],
)
def test_grouped_operations_acknowledge_and_close(
    monkeypatch, path, method, args, output
):
    phone = NS(close=AsyncMock())
    result_value = (
        "de-DE"
        if method == "get_language"
        else "Europe/Berlin" if method == "get_timezone" else None
    )
    operation = AsyncMock(return_value=result_value)
    setattr(phone, method, operation)
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=phone))
    target = CLOUD_ID if method == "set_clipboard" else "serial"
    result = CliRunner().invoke(device, [*path, "-d", target])
    assert result.exit_code == 0, (result.output, result.exception)
    operation.assert_awaited_once_with(*args)
    phone.close.assert_awaited_once()
    assert output in result.stdout
    if method == "set_clipboard":
        assert "private text" not in result.output


@pytest.mark.parametrize(
    "path",
    [
        ["input", "tap", "1", "2"],
        ["screen", "ui"],
        ["app", "stop", "app.id"],
        ["settings", "language", "get"],
        ["action", "list"],
    ],
)
def test_conflicting_target_never_connects(monkeypatch, path):
    connect = AsyncMock()
    monkeypatch.setattr(selection, "connect_target", connect)
    result = CliRunner().invoke(device, [*path, "first", "-d", "second"])
    assert result.exit_code == 2
    assert "refer to different devices" in result.output
    connect.assert_not_awaited()


def test_operation_failure_closes(monkeypatch):
    phone = NS(close=AsyncMock(), stop_app=AsyncMock(side_effect=ValueError("bad app")))
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=phone))
    result = CliRunner().invoke(device, ["app", "stop", "bad", "-d", "serial"])
    assert result.exit_code == 1
    phone.close.assert_awaited_once()


@pytest.mark.parametrize("suffix", [[], ["-h"], ["--help"]])
def test_clipboard_bare_and_help_never_connect(monkeypatch, suffix):
    connect = AsyncMock()
    monkeypatch.setattr(selection, "connect_target", connect)
    result = CliRunner().invoke(cli, ["device", "input", "clipboard", *suffix])
    assert result.exit_code in {0, 2}
    assert "get" in result.output and "set" in result.output
    connect.assert_not_awaited()


@pytest.mark.parametrize(
    "name",
    [
        "install",
        "uninstall",
        "pull",
        "push",
        "files",
        "rm",
        "open",
        "location",
        "language",
        "timezone",
        "clipboard",
        "actions",
        "call",
        "extensions",
    ],
)
def test_branch_only_flat_paths_removed(name):
    result = CliRunner().invoke(device, [name, "--help"])
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "path",
    [["app"], ["file"], ["input"], ["screen"], ["settings"], ["action"], ["cloud"]],
)
def test_group_packages_match_tree(path):
    module = importlib.import_module("mobilerun.cli.devices." + path[0])
    assert module.__file__.endswith("/__init__.py")
    result = CliRunner().invoke(device, [*path, "--help"])
    assert result.exit_code == 0


def test_every_device_command_module_matches_its_path():
    import click

    def visit(group, path):
        for name, command in group.commands.items():
            if command.hidden:
                continue
            expected = path + "." + name.replace("-", "_")
            assert command.callback.__module__ == expected
            if isinstance(command, click.Group):
                module = importlib.import_module(expected)
                assert module.__file__.endswith("/__init__.py")
                visit(command, expected)

    visit(device, "mobilerun.cli.devices")


@pytest.mark.parametrize("operation", ["status", "stop", "reboot", "terminate"])
def test_cloud_lifecycle_uses_sdk_boundary_without_phone_connect(
    monkeypatch, operation
):
    module = importlib.import_module(f"mobilerun.cli.devices.cloud.{operation}")
    connect = AsyncMock(side_effect=AssertionError("must not open a phone"))
    monkeypatch.setattr(selection, "connect_target", connect)
    boundary = AsyncMock(
        return_value=NS(state="stopped", model_dump_json=lambda: '{"state":"stopped"}')
    )
    monkeypatch.setattr(
        module.devices, "retrieve" if operation == "status" else operation, boundary
    )
    fake_client = object()
    seen = []

    @asynccontextmanager
    async def client(**kwargs):
        seen.append(kwargs)
        yield fake_client
        seen.append("closed")

    monkeypatch.setattr(module, "cloud_client", client)
    monkeypatch.setattr(
        module, "get_settings", lambda: NS(MOBILERUN_BILLING=Billing.NO_COMMERCE)
    )
    args = ["cloud", operation, "-d", CLOUD_ID, "--base-url", "https://example.invalid"]
    if operation in {"terminate", "reboot"}:
        refused = CliRunner().invoke(device, args)
        assert refused.exit_code == 1
        boundary.assert_not_awaited()
        assert not seen
        args.append("--yes")
    result = CliRunner().invoke(device, args)
    assert result.exit_code == 0, (result.output, result.exception)
    boundary.assert_awaited_once_with(
        CLOUD_ID, billing=Billing.NO_COMMERCE, client=fake_client
    )
    assert seen == [
        {"base_url": "https://example.invalid", "billing": Billing.NO_COMMERCE},
        "closed",
    ]
    connect.assert_not_awaited()


@pytest.mark.parametrize(
    "operation", ["status", "resume", "stop", "reboot", "terminate"]
)
def test_cloud_lifecycle_rejects_local_before_client(monkeypatch, operation):
    module = importlib.import_module(f"mobilerun.cli.devices.cloud.{operation}")
    monkeypatch.setattr(
        module, "cloud_client", lambda **kwargs: pytest.fail("must not call cloud")
    )
    result = CliRunner().invoke(device, ["cloud", operation, "-d", "serial"])
    assert result.exit_code == 2
    assert "requires a cloud device ID" in result.output


def test_action_list_json(monkeypatch):
    from mobilerun.core.devices.base import Device

    phone = Device()
    phone.close = AsyncMock()
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=phone))
    result = CliRunner().invoke(device, ["action", "list", "--json", "-d", "serial"])
    assert result.exit_code == 0, result.output
    assert any(entry["name"] == "device.wait" for entry in json.loads(result.stdout))
    phone.close.assert_awaited_once()


@pytest.mark.parametrize("operation", ["start", "stop", "uninstall"])
def test_installed_app_picker(monkeypatch, operation):
    from mobilerun.cli.devices.base import pickers

    phone = NS(
        close=AsyncMock(),
        list_apps=AsyncMock(
            return_value=[{"package": "org.example", "label": "Example"}]
        ),
    )
    method = AsyncMock()
    setattr(phone, operation + "_app", method)
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=phone))
    monkeypatch.setattr(pickers, "is_interactive", lambda: True)
    seen = []

    def pick(label, choices):
        seen.extend(choices)
        return "org.example"

    monkeypatch.setattr(pickers, "select_prompt_async", AsyncMock(side_effect=pick))
    args = ["app", operation, "-d", "serial"]
    if operation == "uninstall":
        args.append("--yes")
    result = CliRunner().invoke(device, args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert seen[0].label == "Example"
    assert seen[-1].value == "__manual__"
    method.assert_awaited_once_with("org.example")
    phone.close.assert_awaited_once()


def test_language_timezone_getters_and_wrappers():
    from mobilerun.clients.cloud import devices
    from mobilerun.core.devices.base import Device
    from mobilerun.core.devices.cloud import CloudDevice
    from mobilerun.core.devices.components.wrapper import DeviceWrapper
    from mobilerun.core.devices.pool import DevicePool
    from mobilerun.errors import DeviceOperationNotSupported

    async def run():
        client = NS(
            devices=NS(
                language=NS(get=AsyncMock(return_value=NS(locale="de-DE"))),
                timezone=NS(get=AsyncMock(return_value=NS(timezone="Europe/Berlin"))),
            )
        )
        phone = CloudDevice(CLOUD_ID, api_key="test")
        owned = phone._owned_client
        phone._client = client
        try:
            assert (
                await devices.get_language(
                    CLOUD_ID, billing=Billing.NO_COMMERCE, client=client
                )
            ).locale == "de-DE"
            assert (
                await devices.get_timezone(
                    CLOUD_ID, billing=Billing.NO_COMMERCE, client=client
                )
            ).timezone == "Europe/Berlin"
            assert await phone.get_language() == "de-DE"
            assert await phone.get_timezone() == "Europe/Berlin"
            pool = DevicePool([DeviceWrapper(phone)])
            assert await pool.get_language() == ["de-DE"]
            assert await pool.get_timezone() == ["Europe/Berlin"]
            for name in ("get_language", "get_timezone"):
                with pytest.raises(DeviceOperationNotSupported, match=name):
                    await getattr(Device(), name)()
        finally:
            await owned.close()

    asyncio.run(run())


def test_cloud_picker_only_fetches_cloud_choices(monkeypatch):
    from mobilerun.cli.base import choices
    from mobilerun.cli.base.prompts import SelectChoice
    from mobilerun.cli.devices.base import options, pickers

    module = importlib.import_module("mobilerun.cli.devices.cloud.status")
    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(pickers, "is_interactive", lambda: True)
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(side_effect=AssertionError("no local discovery")),
    )
    monkeypatch.setattr(
        choices, "cloud_devices", lambda ctx: [SelectChoice(CLOUD_ID, "Stopped phone")]
    )
    monkeypatch.setattr(
        pickers, "select_prompt", lambda label, entries: entries[0].value
    )
    monkeypatch.setattr(
        module.devices, "retrieve", AsyncMock(return_value=NS(state="stopped"))
    )

    @asynccontextmanager
    async def client(**kwargs):
        yield object()

    monkeypatch.setattr(module, "cloud_client", client)
    result = CliRunner().invoke(device, ["cloud", "status"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert CLOUD_ID in result.stdout


def test_file_transfer_progress_is_only_stderr(monkeypatch, tmp_path):
    phone = NS(
        close=AsyncMock(),
        push_file=AsyncMock(return_value=NS(path="/sdcard/file")),
        pull_file=AsyncMock(return_value=tmp_path / "out"),
    )
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=phone))
    local = tmp_path / "in"
    local.write_text("file")
    push = CliRunner().invoke(device, ["file", "push", str(local), "-d", "serial"])
    assert push.exit_code == 0, (push.output, push.exception)
    assert push.stdout == "/sdcard/file\n"
    assert "Pushing to serial" in push.stderr
    pull = CliRunner().invoke(device, ["file", "pull", "/sdcard/file", "-d", "serial"])
    assert pull.exit_code == 0, (pull.output, pull.exception)
    assert pull.stdout == f"{tmp_path / 'out'}\n"
    assert "Pulling from serial" in pull.stderr


def test_failed_connect_closes_constructed_device(monkeypatch):
    phone = NS(
        connect=AsyncMock(side_effect=RuntimeError("offline")), close=AsyncMock()
    )
    monkeypatch.setattr(selection, "AndroidDevice", lambda **kwargs: phone)
    result = CliRunner().invoke(device, ["input", "tap", "1", "2", "-d", "serial"])
    assert result.exit_code == 1
    phone.close.assert_awaited_once()


@pytest.mark.parametrize("flag", ["--device", "-d"])
def test_agent_run_device_option_reaches_runner(monkeypatch, flag):
    pytest.importorskip("mobilerun_agents", reason="requires mobilerun-agents add-on")
    module = importlib.import_module("mobilerun_agents.cli.agent.run")
    phone = NS(close=AsyncMock())
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=phone))
    execute = AsyncMock(return_value=NS(answer="done", success=True))
    monkeypatch.setattr(module, "AgentOperator", lambda **kwargs: NS(execute=execute))
    monkeypatch.setattr(module.ConfigLoader, "load", lambda _: NS())
    result = CliRunner().invoke(cli, ["agent", "run", "Open settings", flag, "serial"])
    assert result.exit_code == 0, (result.output, result.exception)
    execute.assert_awaited_once_with(phone, "Open settings", max_steps=None)
    assert selection.connect_target.await_args.args == ("serial",)
    assert selection.connect_target.await_args.kwargs["allow_auto_setup"] is True
    phone.close.assert_awaited_once()


@pytest.mark.parametrize("flag", ["--device", "-d"])
def test_agent_replay_device_option_reaches_replay(monkeypatch, tmp_path, flag):
    pytest.importorskip("mobilerun_agents", reason="requires mobilerun-agents add-on")
    module = importlib.import_module("mobilerun_agents.cli.macro_agent.replay")
    monkeypatch.setattr(module.ConfigLoader, "load", lambda _: NS())
    received = []
    monkeypatch.setattr(
        module, "run_replay", lambda operator, **kwargs: received.append(kwargs)
    )
    path = tmp_path / "macro.json"
    path.write_text("{}")
    result = CliRunner().invoke(
        cli, ["macro-agent", "replay", str(path), flag, "serial"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert received[0]["target"].identifier == "serial"
    assert received[0]["target"].kind == "android"
    assert received[0]["target"].allow_auto_setup is True


def test_action_picker_keeps_manual_entry(monkeypatch):
    from mobilerun.cli.devices.base import pickers
    from mobilerun.core.devices.base import Device

    module = importlib.import_module("mobilerun.cli.devices.action.call")
    phone = Device()
    phone.close = AsyncMock()
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=phone))
    monkeypatch.setattr(module, "is_interactive", lambda: True)
    monkeypatch.setattr(pickers, "is_interactive", lambda: True)
    seen = []

    def pick(label, choices):
        seen.extend(choice.value for choice in choices)
        return "__manual__"

    monkeypatch.setattr(pickers, "select_prompt_async", AsyncMock(side_effect=pick))
    monkeypatch.setattr(
        pickers, "text_prompt_async", AsyncMock(return_value="device.wait")
    )
    result = CliRunner().invoke(
        device, ["action", "call", "--arg", "seconds=0", "-d", "serial"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert {"device.wait", "__manual__"} <= set(seen)
    phone.close.assert_awaited_once()


def test_macro_dry_run_keeps_transport_flags_without_device(monkeypatch, tmp_path):
    from mobilerun.cli.devices.base import options

    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(side_effect=AssertionError("must not discover")),
    )
    connect = AsyncMock(side_effect=AssertionError("must not connect"))
    monkeypatch.setattr(selection, "connect_target", connect)
    path = tmp_path / "macro.json"
    path.write_text(json.dumps({"macro_schema_version": "2.0", "actions": []}))
    result = CliRunner().invoke(
        cli, ["macro", "replay", str(path), "--dry-run", "--android", "--tcp"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    connect.assert_not_awaited()


def test_device_help_lists_lifecycle_and_actions():
    result = CliRunner().invoke(device, ["--help"])
    assert result.exit_code == 0
    for command in (
        "list",
        "connect",
        "disconnect",
        "ping",
        "setup",
        "input",
        "app",
        "screen",
        "file",
        "settings",
        "action",
        "cloud",
    ):
        assert command in result.output


def test_device_discovery_failure_and_exclusive_flags(monkeypatch):
    from mobilerun.cli.devices.base import options

    monkeypatch.setattr(options, "discover_devices", AsyncMock(return_value=[]))
    runner = CliRunner()
    expected = "No devices found."
    missing = runner.invoke(device, ["input", "tap", "1", "1"])
    assert missing.exit_code == 1
    assert expected in missing.output
    multiple = runner.invoke(
        device, ["input", "tap", "1", "1", "target", "--android", "--ios"]
    )
    assert multiple.exit_code == 2
    assert "Choose at most one device type flag" in multiple.output


@pytest.mark.parametrize(
    ("args", "method", "positional", "keyword"),
    [
        (["input", "tap", "10", "20"], "tap", (10, 20), {}),
        (
            ["input", "long-press", "1", "2"],
            "swipe",
            (1, 2, 1, 2, 1000),
            {},
        ),
        (
            ["input", "swipe", "1", "2", "3", "4", "--duration", "0.5"],
            "swipe",
            (1, 2, 3, 4),
            {"duration_ms": 500},
        ),
        (
            ["input", "type", "hello", "--clear"],
            "input_text",
            ("hello", True),
            {},
        ),
        (["input", "press", "home"], "press_button", ("home",), {}),
        (["app", "start", "example.app"], "start_app", ("example.app",), {}),
        (
            ["app", "list", "--system"],
            "list_apps",
            (),
            {"include_system": True},
        ),
        (["screen", "screenshot"], "screenshot", (), {}),
    ],
    ids=["tap", "long-press", "swipe", "type", "press", "start", "apps", "screenshot"],
)
def test_action_commands_call_expected_driver_methods(
    monkeypatch, tmp_path, args, method, positional, keyword
):
    module = importlib.import_module(
        f"mobilerun.cli.devices.{args[0]}.{args[1].replace('-', '_')}"
    )
    driver = AsyncMock()
    driver.list_apps.return_value = []
    driver.screenshot.return_value = b"png"
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=driver))
    if method == "screenshot":
        monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path))
    result = CliRunner().invoke(device, [*args, "serial", "--android"])
    assert result.exit_code == 0, (args, result.output, result.exception)
    getattr(driver, method).assert_awaited_once_with(*positional, **keyword)
    driver.close.assert_awaited_once_with()


def test_ui_prints_formatted_state(monkeypatch):
    ui = importlib.import_module("mobilerun.cli.devices.screen.ui")

    driver = AsyncMock()
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=driver))

    monkeypatch.setattr(ui, "log_progress_to_stderr", lambda: None)
    provider = NS(
        get_state=AsyncMock(return_value=NS(formatted_text="tree", phone_state=None))
    )
    driver._create_state_provider = lambda options: provider
    result = CliRunner().invoke(device, ["screen", "ui", "serial", "--android"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "  tree\n"


def test_tap_config_is_passed_to_factory_and_flags_override(monkeypatch, tmp_path):
    from mobilerun.config.migrations import CURRENT_VERSION

    config = tmp_path / "device.yaml"
    config.write_text(
        f"_version: {CURRENT_VERSION}\ndevice:\n  use_tcp: true\n  platform: ios\n  auth_token: example\n  portal_mode: disabled\n  auto_setup: false\n"
    )
    factory = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(selection, "connect_target", factory)

    result = CliRunner().invoke(
        device,
        [
            "input",
            "tap",
            "1",
            "2",
            "serial",
            "--config",
            str(config),
            "--android",
            "--no-tcp",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    args = factory.await_args.kwargs
    assert args["kind"] == "android"
    assert args["tcp"] is False
    assert args["config"].device.auth_token == "example"
    assert args["config"].device.portal_mode == "disabled"
    assert args["config"].device.auto_setup is False


def test_tap_loads_default_config_and_uses_tcp(monkeypatch):
    from mobilerun.config.loader import ConfigLoader
    from mobilerun.config.user_config import MobileConfig

    config = MobileConfig()
    config.device.use_tcp = True
    paths = []
    monkeypatch.setattr(ConfigLoader, "load", lambda path: paths.append(path) or config)
    factory = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(selection, "connect_target", factory)

    result = CliRunner().invoke(device, ["input", "tap", "1", "2", "serial"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert paths == [None]
    assert factory.await_args.kwargs["tcp"] is True
    assert factory.await_args.kwargs["allow_auto_setup"] is False


def test_tap_config_platform_is_used_for_url(monkeypatch, tmp_path):
    from mobilerun.config.migrations import CURRENT_VERSION

    config = tmp_path / "device.yaml"
    config.write_text(f"_version: {CURRENT_VERSION}\ndevice:\n  platform: ios\n")
    factory = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(selection, "connect_target", factory)

    result = CliRunner().invoke(
        device, ["input", "tap", "1", "2", "http://phone", "--config", str(config)]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert factory.await_args.kwargs["kind"] == "ios"


def test_device_discovery_and_default_resolution(monkeypatch, tmp_path):
    from mobilerun.clients.portal import discovery
    from mobilerun.config.migrations import CURRENT_VERSION

    monkeypatch.setattr(
        discovery.adb,
        "list",
        AsyncMock(return_value=[NS(serial="emulator", state="device")]),
    )
    factory = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(selection, "connect_target", factory)

    runner = CliRunner()
    result = runner.invoke(device, ["input", "tap", "1", "2"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert factory.await_args.args[0] == "emulator"
    assert "Using emulator" in result.stderr
    monkeypatch.setenv("MOBILERUN_DEFAULT_DEVICE", "default")
    result = runner.invoke(device, ["input", "tap", "1", "2"])
    assert result.exit_code == 0, result.output
    assert factory.await_args.args[0] == "emulator"
    monkeypatch.setattr(
        discovery.adb,
        "list",
        AsyncMock(return_value=[NS(serial="configured", state="device")]),
    )
    config = tmp_path / "device.yaml"
    config.write_text(f"_version: {CURRENT_VERSION}\ndevice:\n  serial: configured\n")
    result = runner.invoke(device, ["input", "tap", "1", "2", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert factory.await_args.args[0] == "configured"
    result = runner.invoke(
        device, ["input", "tap", "1", "2", "explicit", "--config", str(config)]
    )
    assert result.exit_code == 0, result.output
    assert factory.await_args.args[0] == "explicit"
    assert "Using device:" not in result.stderr


def test_device_validates_saved_serial_against_discovery(monkeypatch):
    from mobilerun.cli.devices.base import options

    config = options.ConfigLoader.load(None)
    config.device.serial = "saved"
    monkeypatch.setattr(options.ConfigLoader, "load", lambda path: config)
    from mobilerun.cli.devices.base.discovery import DiscoveredDevice

    discovery = AsyncMock(
        return_value=[DiscoveredDevice("saved", "local", None, "device", None)]
    )
    monkeypatch.setattr(options, "discover_devices", discovery)
    factory = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(selection, "connect_target", factory)

    result = CliRunner().invoke(device, ["input", "tap", "1", "2"])
    assert result.exit_code == 0, result.output
    assert factory.await_args.args[0] == "saved"
    discovery.assert_awaited_once()


def test_multiple_devices_noninteractive_error(monkeypatch):
    from mobilerun.cli.devices.base import options
    from mobilerun.cli.devices.base.discovery import DiscoveredDevice

    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(
            return_value=[
                DiscoveredDevice("a", "local", None, "device", None),
                DiscoveredDevice("b", "cloud", None, "ready", "android"),
            ]
        ),
    )
    result = CliRunner().invoke(device, ["input", "tap", "1", "2"])
    assert result.exit_code == 1, (result.output, result.exception)
    assert "a (local)" in result.output
    assert "b (cloud)" in result.output


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("serial", "android"),
        ("192.168.1.20:5555", "android"),
        ("emulator-5554", "android"),
        ("123e4567-e89b-12d3-a456-426614174000", "cloud"),
    ],
)
def test_detect_device_kind(target, expected):
    from mobilerun.cli.devices.base.selection import detect_device_kind

    assert detect_device_kind(target) == expected


def test_detect_device_kind_rejects_ambiguous_http_url():
    from mobilerun.cli.devices.base.selection import detect_device_kind

    with pytest.raises(ValueError, match="URL target: add -i/--ios"):
        detect_device_kind("http://127.0.0.1:8000")


def test_connect_target_builds_android_and_closes_android_only(monkeypatch):
    made = []

    class FakeAndroid:
        device = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            made.append(self)

        async def connect(self):
            pass

    monkeypatch.setattr(selection, "AndroidDevice", FakeAndroid)
    device = asyncio.run(selection.connect_target("serial", tcp=True))
    assert isinstance(device, FakeAndroid)
    assert made[0].kwargs == {"serial": "serial", "use_tcp": True}


def test_create_android_url_uses_portal_http_driver(monkeypatch):
    class FakeDriver:
        async def connect(self):
            pass

    driver = FakeDriver()

    def make_driver(url, token):
        return driver

    monkeypatch.setattr(selection, "create_android_portal_http_driver", make_driver)
    monkeypatch.setattr(
        selection.ConfigLoader,
        "load",
        lambda _: type(
            "C",
            (),
            {
                "device": type(
                    "D", (), {"resolve_auth_token": lambda self, platform: "token"}
                )()
            },
        )(),
    )
    monkeypatch.setattr(
        selection, "AndroidDevice", lambda **kwargs: pytest.fail("ADB used")
    )

    result = asyncio.run(selection.connect_target("http://portal.test", kind="android"))
    from typing import cast

    from mobilerun.core.devices.components.wrapper import DriverDevice

    assert cast(DriverDevice, result)._driver is driver
    from mobilerun.core.state.android import AndroidStateProvider
    from mobilerun.core.state.options import StateProviderOptions

    assert isinstance(
        result._create_state_provider(StateProviderOptions()), AndroidStateProvider
    )


def test_connect_target_passes_android_configuration(monkeypatch):
    driver = NS(connect=AsyncMock())

    def constructor(**kwargs):
        seen.update(kwargs)
        return driver

    seen = {}
    monkeypatch.setattr(selection, "AndroidDevice", constructor)
    config = NS(device=NS(auto_setup=False, portal_mode="disabled"))
    assert (
        asyncio.run(selection.connect_target("serial", tcp=True, config=config))
        is driver
    )
    assert seen == {"serial": "serial", "use_tcp": True, "portal_mode": "disabled"}
    driver.connect.assert_awaited_once()


def test_connect_target_constructs_cloud_and_visual_remote(monkeypatch):
    from mobilerun.core.devices import cloud

    driver = NS(connect=AsyncMock())
    constructor = Mock(return_value=driver)
    monkeypatch.setattr(cloud, "CloudDevice", constructor)
    assert (
        asyncio.run(
            selection.connect_target("id", kind="cloud", base_url="https://cloud")
        )
        is driver
    )
    constructor.assert_called_once_with(device_id="id", base_url="https://cloud")
    constructor = Mock(return_value=driver)
    monkeypatch.setattr(selection, "RemoteDevice", constructor)
    config = NS(device=NS(device_id="remote-id"))
    assert (
        asyncio.run(
            selection.connect_target(
                "http://remote", kind="visual_remote", config=config
            )
        )
        is driver
    )
    constructor.assert_called_once_with(url="http://remote", device_id="remote-id")


def test_connect_target_constructs_ios_with_token(monkeypatch):
    driver = NS(connect=AsyncMock())
    create = Mock(return_value=driver)
    monkeypatch.setattr(selection, "IOSDevice", create)
    config = NS(device=NS(resolve_auth_token=lambda platform: "token"))
    result = asyncio.run(
        selection.connect_target("http://ios", kind="ios", config=config)
    )
    assert result is driver
    create.assert_called_once_with("http://ios", token="token")
    driver.connect.assert_awaited_once()
