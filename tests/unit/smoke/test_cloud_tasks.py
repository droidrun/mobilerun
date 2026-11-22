"""Cloud tasks spend credits; select explicitly and keep the step limit low."""

import asyncio
import time

import pytest
from conftest import CLOUD_ID
from pydantic import BaseModel

from mobilerun.clients.cloud import tasks

pytestmark = pytest.mark.smoke_cloud


class VisibleScreen(BaseModel):
    screen: str


def test_cloud_do_and_expect(cloud_ready):
    from mobilerun.core.devices.cloud import CloudDevice

    async def run():
        device = CloudDevice(device_id=CLOUD_ID)
        await device.connect()
        try:
            start = time.monotonic()
            result = await device.do("Open the Settings app", max_steps=3)
            print(
                f"do task_id={result.task_id} duration={time.monotonic() - start:.1f}s success={result.success}"
            )
            assert result.success
            start = time.monotonic()
            result = await device.expect("The Settings app is open")
            print(
                f"expect task_id={result.task.task_id} duration={time.monotonic() - start:.1f}s passed={result.passed}"
            )
            assert result.passed
        finally:
            await device.press_button("home")

    asyncio.run(run())


def test_cloud_typed_output(cloud_ready):
    from mobilerun.core.devices.cloud import CloudDevice

    async def run():
        device = CloudDevice(device_id=CLOUD_ID)
        await device.connect()
        try:
            device._task_started_callback = lambda task_id: print(
                f"typed task_id={task_id}", flush=True
            )
            started = time.monotonic()
            result = await device.do(
                "Describe the current screen briefly", output=VisibleScreen, max_steps=1
            )
            print(
                f"typed duration={time.monotonic() - started:.1f}s success={result.success}"
            )
            assert result.success and isinstance(result.output, VisibleScreen)
            assert (
                await tasks.trajectory(
                    device._cloud_client, result.task_id, billing="included"
                )
                is not None
            )
            assert (
                await tasks.screenshots(
                    device._cloud_client, result.task_id, billing="included"
                )
                is not None
            )
        finally:
            await device.press_button("home")

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancel"])
def test_cloud_do_timeout_stops_task(cloud_ready, cancel):
    from mobilerun.core.devices.cloud import CloudDevice
    from mobilerun.errors import CloudTaskTimeout

    async def run():
        device = CloudDevice(device_id=CLOUD_ID)
        await device.connect()
        task_ids = []
        started = asyncio.Event()

        def task_started(task_id):
            task_ids.append(task_id)
            print(f"task_id={task_id}", flush=True)
            started.set()

        device._task_started_callback = task_started
        try:
            pending = asyncio.create_task(
                device.do(
                    "Open the Settings app, then open About phone, then scroll to the bottom",
                    timeout=30 if cancel else 5,
                )
            )
            if cancel:
                await asyncio.wait_for(started.wait(), 20)
                pending.cancel()
            with pytest.raises(asyncio.CancelledError if cancel else CloudTaskTimeout):
                await pending
            assert task_ids
            for _ in range(20):
                status = await device._client.tasks.get_status(task_ids[0])
                print(f"task_id={task_ids[0]} status={status.status}", flush=True)
                if status.status in {"cancelled", "stopped", "failed", "completed"}:
                    break
                await asyncio.sleep(1)
            assert status.status in {"cancelled", "stopped", "failed"}
        finally:
            await device.press_button("home")

    asyncio.run(run())
