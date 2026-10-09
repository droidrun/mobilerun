import asyncio
from types import SimpleNamespace

import pytest
from mobilerun_core_local.driver.android.portal import (
    PORTAL_PACKAGE_NAME,
    portal_ime_id,
)

from mobilerun.cli import main as cli_main
from mobilerun.config_manager.config_manager import (
    DeviceConfig,
    MobileConfig,
    TelemetryConfig,
)

DISABLE_IME = f"ime disable {portal_ime_id(PORTAL_PACKAGE_NAME)}"


class FakeAdbDevice:
    def __init__(self, serial):
        self.serial = serial
        self.commands = []

    async def shell(self, command):
        self.commands.append(command)
        return ""


@pytest.fixture
def fake_adb(monkeypatch):
    devices = []

    async def device(serial=None):
        devices.append(FakeAdbDevice(serial))
        return devices[-1]

    monkeypatch.setattr(cli_main.adb, "device", device)
    return devices


def _serials(devices):
    return [device.serial for device in devices]


def _run_cli(monkeypatch, config, agent_factory, **kwargs):
    monkeypatch.setattr(cli_main.ConfigLoader, "load", lambda _: config)
    monkeypatch.setattr(cli_main, "_setup_cli_logging", lambda _: None)
    monkeypatch.setattr(cli_main, "print_telemetry_message", lambda **_: None)
    monkeypatch.setattr(cli_main.console, "print", lambda *a, **kw: None)
    monkeypatch.setattr(cli_main, "MobileAgent", agent_factory)
    return asyncio.run(cli_main.run_command("Open settings", debug=False, **kwargs))


class FakeHandler:
    async def stream_events(self):
        if False:
            yield None

    def __await__(self):
        async def done():
            return SimpleNamespace(success=True)

        return done().__await__()


def _agent_resolving(serial):
    class FakeAgent:
        def __init__(self, **kwargs):
            self.android_serial = None

        def run(self):
            self.android_serial = serial
            return FakeHandler()

    return FakeAgent


def test_run_cleans_up_on_agent_resolved_serial(monkeypatch, fake_adb):
    config = MobileConfig()

    assert _run_cli(monkeypatch, config, _agent_resolving("emulator-5556")) is True

    assert _serials(fake_adb) == ["emulator-5556"]
    assert fake_adb[0].commands == [DISABLE_IME]
    assert config.device.serial is None


def test_run_cleans_up_on_explicit_serial(monkeypatch, fake_adb):
    config = MobileConfig()

    assert (
        _run_cli(
            monkeypatch,
            config,
            _agent_resolving("emulator-5558"),
            device="emulator-5558",
        )
        is True
    )

    assert _serials(fake_adb) == ["emulator-5558"]


@pytest.mark.parametrize("device", [None, "emulator-5558"])
def test_run_falls_back_to_config_serial_without_agent(monkeypatch, fake_adb, device):
    def failing_agent(**kwargs):
        raise RuntimeError("agent init failed")

    assert _run_cli(monkeypatch, MobileConfig(), failing_agent, device=device) is False

    assert _serials(fake_adb) == [device]


def test_cleanup_prefers_resolved_serial_over_config_serial(fake_adb):
    config = MobileConfig(device=DeviceConfig(serial="emulator-5558"))

    asyncio.run(cli_main._cleanup_android_keyboard(config))
    asyncio.run(cli_main._cleanup_android_keyboard(config, "emulator-5560"))

    assert _serials(fake_adb) == ["emulator-5558", "emulator-5560"]


@pytest.mark.parametrize(
    "device_config",
    [
        DeviceConfig(platform="ios"),
        DeviceConfig(control_backend="visual-remote"),
    ],
)
def test_cleanup_skips_non_android_backends(fake_adb, device_config):
    config = MobileConfig(device=device_config)

    asyncio.run(cli_main._cleanup_android_keyboard(config, "emulator-5556"))

    assert fake_adb == []


def _agent_module(monkeypatch, online_serial):
    from async_adbutils import adb

    from mobilerun.agent.droid import droid_agent as agent_module

    async def list_devices():
        return [SimpleNamespace(serial=online_serial, state="device")]

    async def device(serial=None):
        return FakeAdbDevice(serial)

    monkeypatch.delenv("ANDROID_SERIAL", raising=False)
    monkeypatch.setattr(agent_module, "setup_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(
        agent_module.MobileAgent, "_configure_default_logging", lambda *a, **kw: None
    )
    monkeypatch.setattr(agent_module, "capture", lambda *a, **kw: None)
    monkeypatch.setattr(adb, "list", list_devices)
    monkeypatch.setattr(adb, "device", device)
    return agent_module


class FakeContext:
    def __init__(self):
        self.store = self

    def write_event_to_stream(self, event):
        return None

    async def set(self, key, value):
        setattr(self, key, value)

    async def get(self, key, default=None):
        return getattr(self, key, default)


class ConnectFailed(Exception):
    pass


@pytest.mark.parametrize(
    ("serial", "expected"),
    [(None, "emulator-5556"), ("emulator-5558", "emulator-5558")],
)
def test_agent_records_android_serial(monkeypatch, serial, expected):
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.workflow import StartEvent

    agent_module = _agent_module(monkeypatch, "emulator-5556")
    constructed = []

    class FakeAndroidDriver:
        def __init__(self, serial, **kwargs):
            constructed.append(serial)

        async def connect(self):
            raise ConnectFailed

    monkeypatch.setattr(agent_module, "AndroidDriver", FakeAndroidDriver)

    config = MobileConfig(
        device=DeviceConfig(serial=serial, auto_setup=False),
        telemetry=TelemetryConfig(enabled=False),
    )
    agent = agent_module.MobileAgent("Open settings", config=config, llms=MockLLM())
    assert agent.android_serial is None

    with pytest.raises(ConnectFailed):
        asyncio.run(agent.start_handler(FakeContext(), StartEvent()))

    assert constructed == [expected]
    assert agent.android_serial == expected
    assert config.device.serial == serial


def test_external_agent_records_android_serial(monkeypatch):
    from llama_index.core.workflow import StartEvent

    agent_module = _agent_module(monkeypatch, "emulator-5556")
    used = []

    async def run_external(device, **kwargs):
        used.append(device.serial)
        return {"success": True, "reason": "done"}

    monkeypatch.setattr(
        agent_module,
        "load_agent",
        lambda name: {"config": {}, "run": run_external},
    )

    config = MobileConfig(telemetry=TelemetryConfig(enabled=False))
    config.agent.name = "external-agent"
    agent = agent_module.MobileAgent("Open settings", config=config)

    asyncio.run(agent.start_handler(FakeContext(), StartEvent()))

    assert used == ["emulator-5556"]
    assert agent.android_serial == "emulator-5556"
    assert config.device.serial is None
