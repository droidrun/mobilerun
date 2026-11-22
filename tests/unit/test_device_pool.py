import asyncio

import pytest

from mobilerun.core.devices import (
    AndroidDevice,
    DevicePool,
    RecordingDevice,
    StealthDevice,
)
from mobilerun.core.devices.base import Device as _FrameworkDevice
from mobilerun.core.operator import Operator
from mobilerun.models.results import ExecutionResult, ExpectationVerdict


class FakeDevice(_FrameworkDevice):
    platform = "fake"
    supported_operations = {"tap"}
    needs_initialization = False

    def __init__(self, value, gate=None, error=None):
        super().__init__()
        self.value = value
        self._operator = None
        self.gate = gate
        self.error = error
        self.finished = False
        self.calls = []

    @property
    def state(self):
        return self

    async def screenshot(self, hide_overlay=True):
        if self.gate:
            await self.gate(self.value)
        self.finished = True
        if self.error:
            raise self.error
        return self.value

    async def _ensure_connected(self):
        self.calls.append("connected")

    async def connect(self):
        await self._ensure_connected()

    async def close(self):
        pass

    async def refresh(self):
        return self.value

    async def do(self, instruction, **kwargs):
        return instruction, kwargs

    async def expect(self, condition, **kwargs):
        return condition, kwargs


def test_empty_and_invalid_concurrency():
    with pytest.raises(ValueError):
        DevicePool([])
    with pytest.raises(ValueError):
        DevicePool([FakeDevice(1)], concurrency=0)


def test_concurrent_order_and_properties():
    async def run():
        ready = asyncio.Event()
        seen = []

        async def gate(value):
            seen.append(value)
            if len(seen) == 2:
                ready.set()
            await ready.wait()

        devices = [FakeDevice(1, gate), FakeDevice(2, gate)]
        pool = DevicePool(devices)
        assert await asyncio.wait_for(pool.screenshot(), 1) == [1, 2]
        assert len(pool) == 2 and tuple(pool) == pool.devices == tuple(devices)
        assert pool.platform == ["fake", "fake"]
        assert pool.supported_operations == [{"tap"}, {"tap"}]
        assert await pool.state.refresh() == [1, 2]

    asyncio.run(run())


def test_errors_wait_and_annotate():
    async def run():
        failure = ValueError("bad")
        devices = [FakeDevice(1, error=failure), FakeDevice(2)]
        with pytest.raises(ExceptionGroup) as caught:
            await DevicePool(devices).screenshot()
        assert all(device.finished for device in devices)
        assert caught.value.exceptions == (failure,)
        assert "device 0: FakeDevice" in failure.__notes__
        assert "1 of 2 devices failed: screenshot" in str(caught.value)

    asyncio.run(run())


def test_operator_forwarding_and_context():
    async def run():
        devices = [FakeDevice(1), FakeDevice(2)]
        operator = object()
        async with DevicePool(devices) as pool:
            assert (
                await pool.do("go", output=int, max_steps=2, operator=operator)
                == [("go", {"output": int, "max_steps": 2, "operator": operator})] * 2
            )
            assert (
                await pool.expect("yes", operator=operator)
                == [("yes", {"operator": operator})] * 2
            )
        assert all(device.calls == ["connected"] for device in devices)

    asyncio.run(run())


def test_concurrency_limit():
    async def run():
        active = peak = 0

        async def gate(value):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

        pool = DevicePool([FakeDevice(i, gate) for i in range(5)], concurrency=2)
        assert await pool.screenshot() == list(range(5))
        assert peak == 2

    asyncio.run(run())


def test_context_closes_supported_devices():
    class Closable(FakeDevice):
        async def close(self):
            self.calls.append("closed")

    async def run():
        devices = [Closable(1), FakeDevice(2)]
        async with DevicePool(devices):
            pass
        assert devices[0].calls == ["connected", "closed"]
        assert devices[1].calls == ["connected"]

    asyncio.run(run())


class FakeOperator(Operator):
    def __init__(self):
        self.calls = []

    async def execute(self, device, instruction, *, output=None, max_steps=None):
        self.calls.append((device, instruction, output, max_steps))
        verdict = ExpectationVerdict(passed=True, reason="visible") if output else None
        return ExecutionResult(success=True, output=verdict, device="fake")


def test_device_expect_with_per_call_operator():
    device = AndroidDevice(serial="test", portal_mode="disabled")
    operator = FakeOperator()
    assert device._operator is None
    assert asyncio.run(device.expect("visible", operator=operator)).passed
    assert operator.calls[0][0] is device
    assert device._operator is None


@pytest.mark.parametrize("wrapper", [RecordingDevice, StealthDevice])
def test_wrapper_forwards_per_call_operator(wrapper):
    device = AndroidDevice(serial="test", portal_mode="disabled")
    wrapped = wrapper(device)
    operator = FakeOperator()
    assert asyncio.run(wrapped.do("go", operator=operator)).success
    assert asyncio.run(wrapped.expect("visible", operator=operator)).passed
    assert [call[0] for call in operator.calls] == [wrapped, wrapped]
    assert device._operator is None


def test_pool_real_android_devices_forward_per_call_operator(monkeypatch):
    async def _ensure_connected(self):
        pass

    monkeypatch.setattr(AndroidDevice, "_ensure_connected", _ensure_connected)
    devices = [AndroidDevice(serial=str(i), portal_mode="disabled") for i in range(2)]
    operator = FakeOperator()

    async def run():
        async with DevicePool(devices) as pool:
            assert all(
                result.success for result in await pool.do("go", operator=operator)
            )
            assert all(
                result.passed
                for result in await pool.expect("visible", operator=operator)
            )

    asyncio.run(run())
    assert [call[0] for call in operator.calls] == devices + devices
    assert all(device._operator is None for device in devices)


def test_cleanup_failure_does_not_replace_body_exception():
    original = RuntimeError("body failed")

    class BrokenClose(FakeDevice):
        async def close(self):
            raise RuntimeError("cleanup failed")

    async def run():
        async with DevicePool([BrokenClose(1)]):
            raise original

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(run())
    assert caught.value is original
    assert any("cleanup" in note.lower() for note in original.__notes__)


@pytest.mark.parametrize("wrapper", [RecordingDevice, StealthDevice])
@pytest.mark.parametrize("pending", ["_ready", "_pending_avd"])
def test_pool_initializes_wrapped_pending_devices(wrapper, pending):
    class Pending(FakeDevice):
        needs_initialization = True

        def __init__(self):
            super().__init__(1)
            setattr(self, pending, pending == "_pending_avd")

        async def connect(self):
            self.calls.append("initialized")

        async def close(self):
            self.calls.append("closed")

    async def run():
        device = Pending()
        wrapped = wrapper(device)
        async with DevicePool([wrapped]):
            assert device.calls == ["initialized"]
        assert device.calls == ["initialized", "closed"]

    asyncio.run(run())
