import asyncio
from types import SimpleNamespace

import click
import pytest
from async_adbutils import AdbDeviceInfo

from mobilerun.agent.utils import android_device
from mobilerun.agent.utils.android_device import (
    AndroidDeviceSelectionError,
    resolve_android_serial,
)


@pytest.fixture
def adb_devices(monkeypatch):
    monkeypatch.delenv("ANDROID_SERIAL", raising=False)
    devices: list[AdbDeviceInfo] = []
    calls = []

    async def fake_list():
        calls.append("list")
        return list(devices)

    monkeypatch.setattr(android_device.adb, "list", fake_list)
    return SimpleNamespace(devices=devices, calls=calls)


def _set(adb_devices, *entries):
    adb_devices.devices[:] = [
        AdbDeviceInfo(serial=serial, state=state) for serial, state in entries
    ]


def test_explicit_serial_is_used_as_is(adb_devices):
    _set(adb_devices, ("emulator-5554", "device"), ("emulator-5556", "device"))

    assert asyncio.run(resolve_android_serial("192.168.1.5:5555")) == "192.168.1.5:5555"
    assert adb_devices.calls == []


def test_env_serial_is_used(adb_devices, monkeypatch):
    monkeypatch.setenv("ANDROID_SERIAL", "  emulator-5556 ")
    _set(adb_devices, ("emulator-5554", "device"), ("emulator-5556", "device"))

    assert asyncio.run(resolve_android_serial()) == "emulator-5556"
    assert adb_devices.calls == []


def test_explicit_serial_wins_over_env(adb_devices, monkeypatch):
    monkeypatch.setenv("ANDROID_SERIAL", "emulator-5556")

    assert asyncio.run(resolve_android_serial("emulator-5554")) == "emulator-5554"


def test_blank_env_serial_is_ignored(adb_devices, monkeypatch):
    monkeypatch.setenv("ANDROID_SERIAL", "   ")
    _set(adb_devices, ("emulator-5554", "device"))

    assert asyncio.run(resolve_android_serial()) == "emulator-5554"


def test_single_online_device_is_selected(adb_devices):
    _set(adb_devices, ("emulator-5554", "device"))

    assert asyncio.run(resolve_android_serial()) == "emulator-5554"


def test_offline_and_unauthorized_devices_are_skipped(adb_devices):
    _set(
        adb_devices,
        ("emulator-5556", "offline"),
        ("R58M123", "unauthorized"),
        ("emulator-5554", "device"),
    )

    assert asyncio.run(resolve_android_serial()) == "emulator-5554"


def test_no_devices_raises(adb_devices):
    with pytest.raises(AndroidDeviceSelectionError) as exc:
        asyncio.run(resolve_android_serial())

    assert str(exc.value) == "No connected Android devices found."
    assert isinstance(exc.value, ValueError)


def test_only_not_ready_devices_are_reported(adb_devices):
    _set(adb_devices, ("emulator-5556", "offline"), ("R58M123", "unauthorized"))

    with pytest.raises(AndroidDeviceSelectionError) as exc:
        asyncio.run(resolve_android_serial())

    message = str(exc.value)
    assert message.startswith("No connected Android devices found.")
    assert "emulator-5556 (offline)" in message
    assert "R58M123 (unauthorized)" in message


def test_multiple_online_devices_raise_with_serials(adb_devices):
    _set(
        adb_devices,
        ("emulator-5554", "device"),
        ("192.168.1.5:5555", "device"),
        ("emulator-5556", "offline"),
    )

    with pytest.raises(AndroidDeviceSelectionError) as exc:
        asyncio.run(resolve_android_serial())

    message = str(exc.value)
    assert "emulator-5554" in message
    assert "192.168.1.5:5555" in message
    assert "emulator-5556" not in message
    assert "-d/--device" in message
    assert "ANDROID_SERIAL" in message


def test_setup_with_multiple_devices_touches_no_device(
    adb_devices, monkeypatch, capsys
):
    from mobilerun.cli import main

    _set(adb_devices, ("emulator-5554", "device"), ("emulator-5556", "device"))
    device_calls = []

    async def fake_device(serial):
        device_calls.append(serial)
        return object()

    async def fake_setup(*args, **kwargs):
        raise AssertionError("setup_portal must not run")

    monkeypatch.setattr(main.adb, "device", fake_device)
    monkeypatch.setattr(main, "setup_portal", fake_setup)

    asyncio.run(main._setup_portal(path=None, device=None, debug=False))

    assert device_calls == []
    out = capsys.readouterr().out
    assert "emulator-5554" in out
    assert "emulator-5556" in out


def test_setup_uses_android_serial(adb_devices, monkeypatch):
    from mobilerun.cli import main

    monkeypatch.setenv("ANDROID_SERIAL", "emulator-5556")
    _set(adb_devices, ("emulator-5554", "device"), ("emulator-5556", "device"))
    device_calls = []

    async def fake_device(serial):
        device_calls.append(serial)
        return object()

    async def fake_setup(*args, **kwargs):
        return True

    monkeypatch.setattr(main.adb, "device", fake_device)
    monkeypatch.setattr(main, "setup_portal", fake_setup)

    asyncio.run(main._setup_portal(path=None, device=None, debug=False))

    assert device_calls == ["emulator-5556"]


def _device_command_config(monkeypatch):
    from mobilerun.cli import device_commands

    monkeypatch.setattr(
        device_commands.ConfigLoader,
        "load",
        lambda _path: SimpleNamespace(
            device=SimpleNamespace(
                serial=None,
                use_tcp=False,
                platform="android",
                auto_setup=False,
                portal_mode="auto",
            )
        ),
    )
    serials = []

    class FakeAndroidDriver:
        def __init__(self, serial, use_tcp, portal_mode):
            serials.append(serial)

        async def connect(self):
            pass

    monkeypatch.setattr(device_commands, "AndroidDriver", FakeAndroidDriver)
    return device_commands, serials


def test_device_command_with_multiple_devices_raises_click_error(
    adb_devices, monkeypatch
):
    device_commands, serials = _device_command_config(monkeypatch)
    _set(adb_devices, ("emulator-5554", "device"), ("emulator-5556", "device"))

    with pytest.raises(click.ClickException) as exc:
        asyncio.run(device_commands._create_driver(None, None, None, False))

    assert "emulator-5554, emulator-5556" in exc.value.message
    assert serials == []


def test_device_command_uses_single_online_device(adb_devices, monkeypatch):
    device_commands, serials = _device_command_config(monkeypatch)
    _set(adb_devices, ("emulator-5556", "unauthorized"), ("emulator-5554", "device"))

    asyncio.run(device_commands._create_driver(None, None, None, False))

    assert serials == ["emulator-5554"]


@pytest.mark.parametrize("agent_name", ["mobilerun", "stub-external"])
def test_agent_with_multiple_devices_raises_before_driver(
    adb_devices, monkeypatch, agent_name
):
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.workflow import StartEvent

    from mobilerun.agent.droid import droid_agent as agent_module
    from mobilerun.config_manager.config_manager import (
        AgentConfig,
        DeviceConfig,
        MobileConfig,
        TelemetryConfig,
    )

    class FakeContext:
        def __init__(self):
            self.store = self

        def write_event_to_stream(self, event) -> None:
            return None

        async def set(self, key, value) -> None:
            setattr(self, key, value)

        async def get(self, key, default=None):
            return getattr(self, key, default)

    class FakeAndroidDriver:
        def __init__(self, *args, **kwargs):
            raise AssertionError("driver must not be created")

    monkeypatch.setattr(agent_module, "setup_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(
        agent_module.MobileAgent, "_configure_default_logging", lambda *a, **kw: None
    )
    monkeypatch.setattr(agent_module, "capture", lambda *a, **kw: None)

    async def fail_device(*args, **kwargs):
        raise AssertionError("device must not be opened")

    async def fail_run(**kwargs):
        raise AssertionError("external agent must not run")

    monkeypatch.setattr(agent_module, "AndroidDriver", FakeAndroidDriver)
    monkeypatch.setattr(agent_module.adb, "device", fail_device)
    monkeypatch.setattr(
        agent_module, "load_agent", lambda _name: {"config": {}, "run": fail_run}
    )
    _set(adb_devices, ("emulator-5554", "device"), ("emulator-5556", "device"))

    agent = agent_module.MobileAgent(
        "Open settings",
        config=MobileConfig(
            agent=AgentConfig(name=agent_name),
            device=DeviceConfig(auto_setup=False),
            telemetry=TelemetryConfig(enabled=False),
        ),
        llms=MockLLM(),
    )

    with pytest.raises(AndroidDeviceSelectionError, match="emulator-5556"):
        asyncio.run(agent.start_handler(FakeContext(), StartEvent()))


def test_macro_replay_with_multiple_devices_exits_with_error(
    adb_devices, monkeypatch, tmp_path
):
    from click.testing import CliRunner

    from mobilerun.macro import cli as macro_cli_module

    async def fake_replay(*args, **kwargs):
        raise AssertionError("replay must not start")

    monkeypatch.setattr(macro_cli_module, "_replay_async", fake_replay)
    _set(adb_devices, ("emulator-5554", "device"), ("emulator-5556", "device"))

    result = CliRunner().invoke(macro_cli_module.macro_cli, ["replay", str(tmp_path)])

    assert result.exit_code == 1
    assert "emulator-5554, emulator-5556" in result.output


def test_macro_dry_run_with_multiple_devices_continues_without_device(
    adb_devices, monkeypatch, tmp_path
):
    from click.testing import CliRunner

    from mobilerun.macro import cli as macro_cli_module

    devices = []

    async def fake_replay(path, device, *args, **kwargs):
        devices.append(device)

    monkeypatch.setattr(macro_cli_module, "_replay_async", fake_replay)
    _set(adb_devices, ("emulator-5554", "device"), ("emulator-5556", "device"))

    result = CliRunner().invoke(
        macro_cli_module.macro_cli, ["replay", str(tmp_path), "--dry-run"]
    )

    assert result.exit_code == 0
    assert devices == [None]
