from click.testing import CliRunner


def test_device_help_lists_lifecycle_and_actions():
    from mobilerun.cli.devices import device

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


def test_tap_action_calls_driver(monkeypatch):
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices import device
    from mobilerun.cli.devices.base import selection as tap

    driver = AsyncMock()
    monkeypatch.setattr(tap, "connect_target", AsyncMock(return_value=driver))

    result = CliRunner().invoke(
        device, ["input", "tap", "10", "20", "serial", "--android"]
    )
    assert result.exit_code == 0, result.output
    driver.tap.assert_awaited_once_with(10, 20)


def test_device_discovery_failure_and_exclusive_flags(monkeypatch):
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices import device
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


def test_action_commands_call_expected_driver_methods(monkeypatch, tmp_path):
    import importlib
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices import device

    cases = [
        (
            "long_press",
            ["input", "long-press", "1", "2"],
            "swipe",
            (1, 2, 1, 2, 1000),
            {},
        ),
        (
            "swipe",
            ["input", "swipe", "1", "2", "3", "4", "--duration", "0.5"],
            "swipe",
            (1, 2, 3, 4),
            {"duration_ms": 500},
        ),
        (
            "type",
            ["input", "type", "hello", "--clear"],
            "input_text",
            ("hello", True),
            {},
        ),
        ("press", ["input", "press", "home"], "press_button", ("home",), {}),
        ("start", ["app", "start", "example.app"], "start_app", ("example.app",), {}),
        (
            "apps",
            ["app", "list", "--system"],
            "list_apps",
            (),
            {"include_system": True},
        ),
        ("screenshot", ["screen", "screenshot"], "screenshot", (), {}),
    ]
    for module_name, args, method, positional, keyword in cases:
        from mobilerun.cli.devices.base import selection

        path = (
            "screen.screenshot"
            if module_name == "screenshot"
            else (
                "app.list"
                if module_name == "apps"
                else "app.start" if module_name == "start" else f"input.{module_name}"
            )
        )
        module = importlib.import_module(f"mobilerun.cli.devices.{path}")
        driver = AsyncMock()
        driver.list_apps.return_value = []
        driver.screenshot.return_value = b"png"
        monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=driver))

        if module_name == "screenshot":
            monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path))
        result = CliRunner().invoke(device, [*args, "serial", "--android"])
        assert result.exit_code == 0, (args, result.output, result.exception)
        getattr(driver, method).assert_awaited_once_with(*positional, **keyword)
        driver.close.assert_awaited_once_with()


def test_ui_prints_formatted_state(monkeypatch):
    import importlib
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices import device
    from mobilerun.cli.devices.base import selection

    ui = importlib.import_module("mobilerun.cli.devices.screen.ui")

    driver = AsyncMock()
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=driver))

    monkeypatch.setattr(ui, "log_progress_to_stderr", lambda: None)
    provider = SimpleNamespace(
        get_state=AsyncMock(
            return_value=SimpleNamespace(formatted_text="tree", phone_state=None)
        )
    )
    driver._create_state_provider = lambda options: provider
    result = CliRunner().invoke(device, ["screen", "ui", "serial", "--android"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "  tree\n"


def test_tap_config_is_passed_to_factory_and_flags_override(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices import device
    from mobilerun.cli.devices.base import selection as tap
    from mobilerun.config.migrations import CURRENT_VERSION

    config = tmp_path / "device.yaml"
    config.write_text(
        f"_version: {CURRENT_VERSION}\ndevice:\n  use_tcp: true\n  platform: ios\n  auth_token: example\n  portal_mode: disabled\n  auto_setup: false\n"
    )
    factory = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(tap, "connect_target", factory)

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
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices import device
    from mobilerun.cli.devices.base import selection as tap
    from mobilerun.config.loader import ConfigLoader
    from mobilerun.config.user_config import MobileConfig

    config = MobileConfig()
    config.device.use_tcp = True
    paths = []
    monkeypatch.setattr(ConfigLoader, "load", lambda path: paths.append(path) or config)
    factory = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(tap, "connect_target", factory)

    result = CliRunner().invoke(device, ["input", "tap", "1", "2", "serial"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert paths == [None]
    assert factory.await_args.kwargs["tcp"] is True
    assert factory.await_args.kwargs["allow_auto_setup"] is False


def test_tap_config_platform_is_used_for_url(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices import device
    from mobilerun.cli.devices.base import selection as tap
    from mobilerun.config.migrations import CURRENT_VERSION

    config = tmp_path / "device.yaml"
    config.write_text(f"_version: {CURRENT_VERSION}\ndevice:\n  platform: ios\n")
    factory = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(tap, "connect_target", factory)

    result = CliRunner().invoke(
        device, ["input", "tap", "1", "2", "http://phone", "--config", str(config)]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert factory.await_args.kwargs["kind"] == "ios"


def test_device_discovery_and_default_resolution(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices import device
    from mobilerun.cli.devices.base import selection as tap
    from mobilerun.clients.portal import discovery
    from mobilerun.config.migrations import CURRENT_VERSION

    monkeypatch.setattr(
        discovery.adb,
        "list",
        AsyncMock(return_value=[SimpleNamespace(serial="emulator", state="device")]),
    )
    factory = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(tap, "connect_target", factory)

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
        AsyncMock(return_value=[SimpleNamespace(serial="configured", state="device")]),
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
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices import device
    from mobilerun.cli.devices.base import options
    from mobilerun.cli.devices.base import selection as tap

    config = options.ConfigLoader.load(None)
    config.device.serial = "saved"
    monkeypatch.setattr(options.ConfigLoader, "load", lambda path: config)
    from mobilerun.cli.devices.base.discovery import DiscoveredDevice

    discovery = AsyncMock(
        return_value=[DiscoveredDevice("saved", "local", None, None, None)]
    )
    monkeypatch.setattr(options, "discover_devices", discovery)
    factory = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(tap, "connect_target", factory)

    result = CliRunner().invoke(device, ["input", "tap", "1", "2"])
    assert result.exit_code == 0, result.output
    assert factory.await_args.args[0] == "saved"
    discovery.assert_awaited_once()


def test_multiple_devices_noninteractive_error(monkeypatch):
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices import device
    from mobilerun.cli.devices.base import options
    from mobilerun.cli.devices.base.discovery import DiscoveredDevice

    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(
            return_value=[
                DiscoveredDevice("a", "local", None, None, None),
                DiscoveredDevice("b", "cloud", None, "ready", "android"),
            ]
        ),
    )
    result = CliRunner().invoke(device, ["input", "tap", "1", "2"])
    assert result.exit_code == 1, (result.output, result.exception)
    assert "a (local)" in result.output
    assert "b (cloud)" in result.output
