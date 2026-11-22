"""Local emulator lifecycle does not require an Android SDK in unit tests."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.devices.android import AndroidDevice


@pytest.fixture(autouse=True)
def fake_connection(monkeypatch):
    monkeypatch.setattr(AndroidDevice, "_ensure_connected", AsyncMock())


def run_device(device):
    async def ready():
        return await device

    return asyncio.run(ready())


def test_create_reuses_running_avd(monkeypatch):
    from mobilerun.clients.local import emulator

    monkeypatch.setenv("MOBILERUN_ANDROID_DEVICE_TYPE", "configured")
    monkeypatch.setattr(
        emulator, "find_running", AsyncMock(return_value="emulator-5554")
    )
    start = AsyncMock()
    monkeypatch.setattr(emulator, "start_avd", start)
    monkeypatch.setattr(emulator, "wait_boot", AsyncMock())
    device = run_device(AndroidDevice(avd="configured"))
    assert device._serial == "emulator-5554"
    emulator.find_running.assert_awaited_once_with("configured")
    start.assert_not_awaited()


def test_create_starts_explicit_avd_and_terminate(monkeypatch):
    from mobilerun.clients.local import emulator

    monkeypatch.setattr(emulator, "find_running", AsyncMock(return_value=None))
    monkeypatch.setattr(emulator, "list_avds", AsyncMock(return_value=["pixel"]))
    monkeypatch.setattr(emulator, "start_avd", AsyncMock(return_value="emulator-5556"))
    monkeypatch.setattr(emulator, "wait_boot", AsyncMock())
    monkeypatch.setattr(emulator, "stop", AsyncMock())
    device = run_device(AndroidDevice(avd="pixel", headless=True))
    assert device._serial == "emulator-5556"
    asyncio.run(device.terminate())
    emulator.stop.assert_awaited_once_with("emulator-5556")


def test_unknown_avd_lists_available_avds(monkeypatch):
    from mobilerun.clients.local import emulator

    monkeypatch.delenv("MOBILERUN_ANDROID_DEVICE_TYPE", raising=False)
    find_running = AsyncMock(return_value=None)
    monkeypatch.setattr(emulator, "find_running", find_running)
    monkeypatch.setattr(emulator, "list_avds", AsyncMock(return_value=["pixel"]))
    start_avd = AsyncMock()
    monkeypatch.setattr(emulator, "start_avd", start_avd)
    with pytest.raises(ValueError, match="pixel"):
        run_device(AndroidDevice(avd="missing"))
    find_running.assert_awaited_once_with("missing")
    start_avd.assert_not_awaited()


def test_terminate_refuses_physical_device():
    with pytest.raises(ValueError, match="emulator"):
        asyncio.run(AndroidDevice(serial="physical").terminate())


def test_client_lists_avds_and_finds_running(monkeypatch):
    from mobilerun.clients.local import emulator

    async def command(*args):
        if args[-1] == "-list-avds":
            return "pixel\ntablet\n"
        if args[-1] == "devices":
            return "List of devices attached\nemulator-5554\tdevice\nphysical\tdevice\n"
        return "pixel\nOK\n"

    monkeypatch.setattr(emulator, "_command", command)
    assert asyncio.run(emulator.list_avds()) == ["pixel", "tablet"]
    assert asyncio.run(emulator.find_running("pixel")) == "emulator-5554"
    assert asyncio.run(emulator.find_running("other")) is None


def test_client_start_uses_emulator_reported_port(monkeypatch):
    from mobilerun.clients.local import emulator

    async def launch(*args, **kwargs):
        assert "-port" not in args
        assert "-no-window" in args
        report = args[args.index("-report-console") + 1]
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", int(report.split(":")[1])
        )
        writer.write(b"5556")
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        return object()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    assert asyncio.run(emulator.start_avd("pixel", headless=True)) == "emulator-5556"


def test_wait_boot_timeout(monkeypatch):
    from mobilerun.clients.local import emulator

    async def command(*args):
        if args[-1] == "sys.boot_completed":
            await asyncio.sleep(1)
        return "0"

    monkeypatch.setattr(emulator, "_command", command)
    with pytest.raises(TimeoutError, match="did not boot"):
        asyncio.run(emulator.wait_boot("emulator-5554", timeout=0.01))


def test_find_running_uses_adb_property_when_console_unavailable(monkeypatch):
    from mobilerun.clients.local import emulator

    async def command(*args):
        if args[-1] == "devices":
            return "List of devices attached\nemulator-5554\tdevice\n"
        if args[-1] == "name":
            return ""
        return "pixel\n"

    monkeypatch.setattr(emulator, "_command", command)
    assert asyncio.run(emulator.find_running("pixel")) == "emulator-5554"


def test_boot_failure_cleans_up_only_new_emulator(monkeypatch):
    from mobilerun.clients.local import emulator

    monkeypatch.setattr(emulator, "find_running", AsyncMock(return_value=None))
    monkeypatch.setattr(emulator, "list_avds", AsyncMock(return_value=["pixel"]))
    monkeypatch.setattr(emulator, "start_avd", AsyncMock(return_value="emulator-5556"))
    monkeypatch.setattr(
        emulator, "wait_boot", AsyncMock(side_effect=TimeoutError("boot"))
    )
    stop = AsyncMock()
    monkeypatch.setattr(emulator, "stop", stop)
    with pytest.raises(TimeoutError, match="boot"):
        run_device(AndroidDevice(avd="pixel"))
    stop.assert_awaited_once_with("emulator-5556")


def test_stop_kills_started_process_if_adb_unavailable(monkeypatch):
    from mobilerun.clients.local import emulator

    process = type(
        "Process",
        (),
        {"returncode": None, "kill": lambda self: None, "wait": AsyncMock()},
    )()
    monkeypatch.setitem(emulator._processes, "emulator-5556", process)
    monkeypatch.setattr(
        emulator, "_command", AsyncMock(side_effect=RuntimeError("offline"))
    )
    killed = []
    monkeypatch.setattr(process, "kill", lambda: killed.append(True))
    asyncio.run(emulator.stop("emulator-5556"))
    assert killed == [True]
