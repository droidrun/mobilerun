"""Public Python and CLI contracts: exports, imports, device behavior, and command names."""

import asyncio
import importlib.util
import inspect
import subprocess
import sys
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner

from mobilerun import AndroidDevice, RecordingDevice, StealthDevice
from mobilerun.cli.app import cli
from mobilerun.core.devices import Device
from mobilerun.core.devices.components.wrapper import DriverDevice
from mobilerun.core.state import StateProvider, StateProviderOptions, UIState
from mobilerun.errors import DeviceNotReady, DeviceOperationNotSupported

DEVICE_NAMES = {
    "supported_operations",
    "extensions",
    "actions",
    "context",
    "cloud_id",
    "unbind_extension",
    "close",
    "connect",
    "delete_file",
    "do",
    "drag",
    "expect",
    "get_clipboard",
    "get_location",
    "get_language",
    "get_timezone",
    "identifier",
    "input_text",
    "install_apk",
    "install_app",
    "known_codes_redactor",
    "register_sensitive_values",
    "list_apps",
    "list_files",
    "mailbox",
    "number",
    "open_url",
    "platform",
    "recording_allowed",
    "recording_device",
    "press_button",
    "press_key_code",
    "pull_file",
    "push_file",
    "reset_location",
    "screenshot",
    "set_clipboard",
    "set_language",
    "set_location",
    "set_timezone",
    "start_app",
    "state",
    "stop_app",
    "supported_buttons",
    "swipe",
    "tap",
    "tap_element",
    "uninstall_app",
    "wait",
}


def test_exact_device_cloud_and_pool_surfaces():
    from mobilerun import CloudDevice, DevicePool

    assert {n for n in dir(Device) if not n.startswith("_")} == DEVICE_NAMES
    assert {n for n in dir(CloudDevice) if not n.startswith("_")} == DEVICE_NAMES | {
        "info",
        "name",
        "reboot",
        "rename",
        "reset",
        "resume",
        "stop",
        "terminate",
        "wait_ready",
    }
    pool = DevicePool([Device()])
    assert {n for n in dir(pool) if not n.startswith("_")} == (
        DEVICE_NAMES
        - {
            "identifier",
            "pull_file",
            "cloud_id",
            "known_codes_redactor",
            "register_sensitive_values",
            "recording_device",
            "recording_allowed",
        }
    ) | {"devices", "bind_each"}


def test_exact_root_and_core_exports():
    import mobilerun
    import mobilerun.core
    import mobilerun.core.devices

    devices = {
        "Device",
        "AndroidDevice",
        "CloudDevice",
        "IOSDevice",
        "RemoteDevice",
        "DevicePool",
        "RecordingDevice",
        "StealthDevice",
    }
    proxies = {
        "CloudRotatingProxy",
        "CloudFixedProxy",
        "CloudSocks5Proxy",
        "LocalProxy",
    }
    assert set(mobilerun.core.devices.__all__) == devices
    assert set(mobilerun.core.__all__) == devices | proxies | {"Workspace"}
    assert set(mobilerun.__all__) == devices | proxies | {
        "Workspace",
        "__version__",
        "DeviceConfig",
        "LoggingConfig",
        "MobileConfig",
        "TelemetryConfig",
        "TracingConfig",
    }
    for name in mobilerun.__all__:
        assert getattr(mobilerun, name) is not None
    for name in proxies:
        assert getattr(mobilerun, name) is getattr(mobilerun.core, name)
        assert getattr(mobilerun, name) is getattr(mobilerun.core.proxy, name)
    for prefix in ("Rotating", "Fixed", "Socks5", "Http"):
        name = prefix + "Proxy"
        assert not hasattr(mobilerun, name)
        assert not hasattr(mobilerun.core, name)
        assert not hasattr(mobilerun.core.proxy, name)
    for name in {
        "AndroidDriver",
        "DeviceDriver",
        "RecordingDriver",
        "VisualRemoteDevice",
    }:
        assert name not in vars(mobilerun)


def test_device_has_no_public_plumbing():
    removed = {
        "attach_state_provider",
        "attach_state_provider_factory",
        "default_state_provider",
        "use_default_state_provider",
        "attach_operator",
        "default_operator",
        "operator",
        "unwrap",
        "initialize",
        "ensure_connected",
        "needs_initialization",
        "get_apps",
        "list_packages",
        "ui_state",
        "recording",
        "state_recovery",
        "input_coordinate_size",
        "get_ui_tree",
        "get_date",
        "is_cloud",
        "push_workspace_file",
        "supported",
    }
    assert removed.isdisjoint(dir(Device))
    assert {
        "connect",
        "state",
        "supported_operations",
        "tap",
        "tap_element",
        "do",
        "expect",
    } <= set(dir(Device))


def test_legacy_agent_root_export_is_absent():
    import mobilerun

    assert "MobileAgentState" not in mobilerun.__all__
    with pytest.raises(AttributeError):
        _ = mobilerun.MobileAgentState


def test_workspace_methods_are_classmethods():
    from mobilerun import Workspace

    expected = {
        "upload_file",
        "list_files",
        "download_file",
        "delete_file",
        "upload_app",
        "list_apps",
        "get_app",
        "delete_app",
        "push_file",
    }
    assert {name for name in vars(Workspace) if not name.startswith("_")} == expected
    assert all(
        isinstance(inspect.getattr_static(Workspace, name), classmethod)
        for name in expected
    )


@pytest.mark.parametrize("suffix", ["telemetry", "credential_manager"])
def test_unadvertised_module_path_is_absent(suffix):
    assert importlib.util.find_spec("mobilerun." + suffix) is None


def test_legacy_mcp_modules_are_absent():
    assert importlib.util.find_spec("mobilerun.mcp.adapter") is None
    assert importlib.util.find_spec("mobilerun.mcp.client") is None


@pytest.mark.parametrize(
    "name",
    ["droidrun", "mobilerun.agent", "mobilerun.config_manager", "mobilerun.tools"],
)
def test_removed_imports(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_removed_root_export_alias():
    import mobilerun

    with pytest.raises(AttributeError):
        _ = mobilerun.MacroPlayer


def test_removed_base_config_legacy_name():
    from mobilerun.config import user_config

    assert not hasattr(user_config, "DroidConfig")


def test_library_import_does_not_install_logging_handler():
    run_python(
        "import mobilerun, logging; assert not logging.getLogger('mobilerun').handlers"
    )


def run_python(code: str) -> None:
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "statement",
    ["import mobilerun", "import mobilerun.models", "from mobilerun import Workspace"],
)
def test_simple_import_does_not_load_backends(statement: str) -> None:
    run_python(f"""
import sys
{statement}
for prefix in ('mobilerun_sdk', 'adbutils', 'async_adbutils', 'httpx', 'PIL',
               'mobilerun.core.devices.android', 'mobilerun.clients.cloud'):
    loaded = [name for name in sys.modules if name == prefix or name.startswith(prefix + '.')]
    assert not loaded, (prefix, loaded)
""")


def test_public_exports_resolve_to_original_objects() -> None:
    run_python("""
import mobilerun
from importlib import import_module
modules = {
    'Device': 'devices.base', 'AndroidDevice': 'devices.android',
    'CloudDevice': 'devices.cloud', 'IOSDevice': 'devices.ios',
    'RemoteDevice': 'devices.remote', 'DevicePool': 'devices.pool',
    'RecordingDevice': 'devices.recording', 'StealthDevice': 'devices.stealth',
    'Workspace': 'workspace', 'CloudFixedProxy': 'proxy.cloud_fixed',
    'CloudRotatingProxy': 'proxy.cloud_rotating',
    'CloudSocks5Proxy': 'proxy.cloud_socks5', 'LocalProxy': 'proxy.local',
}
assert set(mobilerun._CORE_EXPORTS) == set(modules)
assert all(getattr(mobilerun, name) is not None for name in mobilerun.__all__)
for name, module in modules.items():
    assert getattr(mobilerun, name) is getattr(import_module('mobilerun.core.' + module), name)
assert set(mobilerun.__all__) <= set(dir(mobilerun))
namespace = {}
exec('from mobilerun import *', namespace)
assert all(namespace[name] is getattr(mobilerun, name) for name in mobilerun.__all__)
""")


def test_generic_device_does_not_import_concrete_backends() -> None:
    run_python("""
import sys
from mobilerun import Device
assert Device.__name__ == 'Device'
assert 'mobilerun.core.devices.android' not in sys.modules
assert 'mobilerun.clients.cloud' not in sys.modules
""")


@pytest.mark.parametrize("option", ["--help", "--version"])
def test_root_info_does_not_load_prompt_libraries(option: str) -> None:
    run_python(f"""
import sys
from click.testing import CliRunner
from mobilerun.cli.app import cli
assert CliRunner().invoke(cli, [{option!r}]).exit_code == 0
assert 'InquirerPy' not in sys.modules
assert 'prompt_toolkit' not in sys.modules
""")


def test_prompt_and_path_helpers_live_in_utils():
    prompts = importlib.import_module("mobilerun.utils.prompts")
    paths = importlib.import_module("mobilerun.utils.paths")
    assert (
        prompts.PromptLoader.render_template("Hello {{ name }}", {"name": "world"})
        == "Hello world"
    )
    assert paths.PathResolver.resolve(".").exists()
    assert callable(prompts.PromptLoader.load_prompt)


def test_async_prompt_loader_renders_file(tmp_path):
    from mobilerun.utils.prompts import PromptLoader

    template = tmp_path / "prompt.jinja2"
    template.write_text("Hello {{ name }}", encoding="utf-8")
    assert (
        asyncio.run(PromptLoader.load_prompt(str(template), {"name": "Ada"}))
        == "Hello Ada"
    )


@pytest.mark.parametrize("wrapper", [RecordingDevice, StealthDevice])
def test_explicit_wrapper_factory_replaces_inherited_provider(wrapper):
    inner = Device(state_provider=StateProvider(Device()))
    factory = Mock(side_effect=lambda device, options: StateProvider(device))
    options = StateProviderOptions(vision=True)
    outer = wrapper(inner, state_provider_factory=factory, state_options=options)
    assert outer.state._resolved_provider._device is outer
    factory.assert_called_once_with(outer, options)


@pytest.mark.parametrize("wrapper", [RecordingDevice, StealthDevice])
def test_wrapper_rejects_conflicting_explicit_provider_and_factory(wrapper):
    with pytest.raises(TypeError, match="state_provider or state_provider_factory"):
        wrapper(
            Device(),
            state_provider=StateProvider(Device()),
            state_provider_factory=lambda device, options: StateProvider(device),
        )


def test_state_controller_caches_returned_snapshot():
    snapshot = UIState(
        elements=[],
        formatted_text="ready",
        focused_text="",
        phone_state={},
        screen_width=100,
        screen_height=200,
    )
    provider = StateProvider(Device())
    provider.get_state = AsyncMock(return_value=snapshot)
    phone = Device(state_provider=provider)
    with pytest.raises(DeviceNotReady, match="refresh"):
        _ = phone.state.current
    assert asyncio.run(phone.state.refresh()) is snapshot
    assert phone.state.current is snapshot
    assert {name for name in dir(phone.state) if not name.startswith("_")} == {
        "refresh",
        "current",
        "capabilities",
    }


def test_driver_connect_is_idempotent_and_translates_unsupported():
    raw = Mock()
    raw.connect = AsyncMock(side_effect=NotImplementedError("no connect"))

    def factory(device, options):
        return StateProvider(device)

    phone = DriverDevice(raw, state_provider_factory=factory)
    with pytest.raises(DeviceOperationNotSupported, match="no connect"):
        asyncio.run(phone.connect())
    raw.connect.side_effect = None
    asyncio.run(phone.connect())
    asyncio.run(phone.connect())
    assert raw.connect.await_count == 2


def test_android_constructor_exposes_typed_state_configuration():
    names = inspect.signature(AndroidDevice).parameters
    assert {
        "state_provider",
        "state_provider_factory",
        "state_options",
        "operator",
    } <= names.keys()


def test_ios_cannot_cache_wrong_provider_before_detection(monkeypatch):
    from mobilerun import IOSDevice
    from mobilerun.core.state.android import AndroidStateProvider

    phone = IOSDevice("http://local")
    with pytest.raises(DeviceNotReady):
        _ = phone.state.capabilities
    monkeypatch.setattr(
        "mobilerun.core.devices.ios.detect_ios_portal_kind",
        AsyncMock(return_value="portal-http"),
    )
    monkeypatch.setattr(phone, "_ensure_connected", AsyncMock())
    asyncio.run(phone.connect())
    assert isinstance(phone.state._resolved_provider, AndroidStateProvider)


def test_android_text_boundary_matches_raw_driver_signature(monkeypatch):
    typed = []

    async def input_text(self, text, clear=False):
        typed.append((text, clear))
        return True

    monkeypatch.setattr(
        "mobilerun.core.devices.android.AndroidDriver.input_text", input_text
    )
    phone = AndroidDevice("fake")
    phone._driver.ensure_connected = AsyncMock()
    assert asyncio.run(phone.input_text("hello", clear=True))
    assert typed == [("hello", True)]


@pytest.mark.parametrize(
    "command",
    [
        "run",
        "configure",
        "devices",
        "login",
        "logout",
        "whoami",
        "connect",
        "disconnect",
        "setup",
        "ping",
        "auth",
        "number sms",
        "device screenshot",
        "device ui",
        "device tap",
        "device swipe",
        "device long-press",
        "device type",
        "device press",
        "device apps",
        "device start",
    ],
)
def test_removed_cli_alias(command):
    path = command.split()
    result = CliRunner().invoke(cli, [*path, "--help"])
    assert result.exit_code == 2
    assert f"No such command '{path[-1]}'" in result.output


def test_unknown_alias_suggests_current_command():
    result = CliRunner().invoke(cli, ["devices"])
    assert "Did you mean 'device'?" in result.output


@pytest.mark.parametrize(
    "path",
    [
        ["account", "login"],
        ["account", "logout"],
        ["account", "status"],
        ["device", "list"],
        ["device", "connect"],
        ["device", "disconnect"],
        ["device", "setup"],
        ["device", "ping"],
        ["device", "input", "tap"],
    ],
)
def test_current_command_help(path):
    result = CliRunner().invoke(cli, [*path, "--help"])
    assert result.exit_code == 0, result.output
