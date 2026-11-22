import asyncio
from unittest.mock import AsyncMock

from mobilerun.core.devices.base import Device as DeviceMixin
from mobilerun.core.devices.pool import DevicePool
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.devices.stealth import StealthDevice
from mobilerun.macro.player import MacroPlayer


def test_wait_sleeps_records_and_replays(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    device = DeviceMixin()
    recording = RecordingDevice(device)
    asyncio.run(StealthDevice(recording).wait(2.5))
    sleep.assert_awaited_once_with(2.5)
    assert recording.recorded_actions == [{"action_type": "wait", "duration": 2.5}]
    assert asyncio.run(MacroPlayer(device).replay_action(recording.recorded_actions[0]))
    assert sleep.await_count == 2
    assert sleep.await_args.args == (2.5,)


def test_pool_waits_once_per_device_in_parallel(monkeypatch):
    entered = []

    async def sleep(seconds):
        entered.append(seconds)
        if len(entered) == 2:
            ready.set()
        await ready.wait()

    monkeypatch.setattr(asyncio, "sleep", sleep)

    async def run():
        nonlocal ready
        ready = asyncio.Event()
        return await DevicePool([DeviceMixin(), DeviceMixin()]).wait(3)

    ready = None
    assert asyncio.run(run()) == [None, None]
    assert entered == [3, 3]
