import asyncio
from types import SimpleNamespace

import pytest

from mobilerun.core.devices.base import Device as DeviceMixin
from mobilerun.core.devices.pool import DevicePool
from mobilerun.core.operator import Operator


def test_device_do_forwards_timeout():
    class FakeOperator(Operator):
        async def execute(self, device, instruction, **kwargs):
            return kwargs

    device = DeviceMixin(operator=FakeOperator())
    assert asyncio.run(device.do("go", timeout=3))["timeout"] == 3


def test_pool_forwards_timeout():
    class Device:
        _operator = None

        async def do(self, instruction, **kwargs):
            return kwargs

        async def expect(self, condition, **kwargs):
            return kwargs

    pool = DevicePool([Device()])
    assert asyncio.run(pool.do("go", timeout=3))[0]["timeout"] == 3
    assert asyncio.run(pool.expect("ok", timeout=4))[0]["timeout"] == 4


def test_macro_timeout_cancels_replay(monkeypatch):
    from mobilerun.macro import operator as macro_module

    cancelled = []

    class Player:
        def __init__(self, device, **kwargs):
            pass

        def load_macro_from_file(self, path):
            return {}

        async def replay_macro(self, *args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

    monkeypatch.setattr(macro_module, "MacroPlayer", Player)
    with pytest.raises(TimeoutError):
        asyncio.run(macro_module.MacroOperator().execute(object(), "go", timeout=0.01))
    assert cancelled == [True]


@pytest.mark.parametrize("cancel", [False, True])
def test_cloud_device_do_stops_task_on_timeout_or_cancel(cancel):
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock

    from mobilerun.core.operator.cloud import CloudTaskOperator
    from mobilerun.errors import CloudTaskTimeout

    async def run():
        started = asyncio.Event()

        class Tasks:
            run = AsyncMock(return_value=SimpleNamespace(id="task-1"))
            stop = AsyncMock()

            @property
            def with_streaming_response(self):
                return self

            @asynccontextmanager
            async def attach(self, task_id, **kwargs):
                yield self

            async def iter_lines(self):
                started.set()
                await asyncio.Event().wait()
                yield ""

        tasks = Tasks()

        class FakeCloudDevice(DeviceMixin):
            @property
            def _is_cloud(self):
                return True

        device = FakeCloudDevice(operator=CloudTaskOperator(billing="included"))
        device._task_started_callback = None
        device.device_id = "device-1"
        device._cloud_client = SimpleNamespace(tasks=tasks)
        device._ensure_connected = AsyncMock()
        pending = asyncio.create_task(device.do("go", timeout=10 if cancel else 0.02))
        await asyncio.wait_for(started.wait(), 1)
        if cancel:
            pending.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else CloudTaskTimeout):
            await pending
        tasks.stop.assert_awaited_once_with("task-1")

    asyncio.run(run())
