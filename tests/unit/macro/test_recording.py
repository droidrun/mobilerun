"""Device wrappers record actions and timing for offline macro replay."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from mobilerun_core_local.driver.base import DeviceDriver

from mobilerun.core.credentials import DictCredentials
from mobilerun.core.devices import Device
from mobilerun.core.devices.pool import DevicePool
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.devices.stealth import StealthDevice
from mobilerun.core.extensions import Extension, device_action
from mobilerun.core.state.view import UIStateView
from mobilerun.macro.errors import MacroError
from mobilerun.macro.player import MacroPlayer
from tests.support.devices import MemoryDevice
from tests.support.macro import FakeDevice, write_macro


class TapDevice(Device):
    platform = "Android"
    capabilities = set()
    supported_buttons = set()

    def __init__(self):
        super().__init__()
        self.taps = []

    async def tap(self, x, y):
        self.taps.append((x, y))

    async def _ensure_connected(self):
        pass


def test_recording_device_is_device_and_records_tap_and_reuses_provider():
    inner = TapDevice()
    inner._ui_state_view = UIStateView(inner, inner._create_state_provider)
    provider = object()
    inner._set_state_provider(provider)
    device = RecordingDevice(inner)
    assert isinstance(device, Device)
    assert not isinstance(device, DeviceDriver)
    assert device.platform == "Android"
    assert device.state._resolved_provider is provider

    asyncio.run(device.tap(12, 34))
    assert inner.taps == [(12, 34)]
    assert device.recorded_actions == [{"action_type": "tap", "x": 12, "y": 34}]


def test_stealth_device_is_device():
    device = StealthDevice(TapDevice())
    assert isinstance(device, Device)
    assert not isinstance(device, DeviceDriver)
    assert device.platform == "Android"


def test_tap_element_on_recording_device_records_resolved_tap():
    class State:
        def get_element_coords(self, index):
            assert index == 0
            return 40, 60

    class Provider:
        capabilities = {"element_index"}

        async def get_state(self):
            return State()

    inner = TapDevice()
    device = RecordingDevice(inner, state_provider=Provider())
    asyncio.run(device.state.refresh())
    asyncio.run(device.tap_element(0))
    assert inner.taps == [(40, 60)]
    assert device.recorded_actions == [{"action_type": "tap", "x": 40, "y": 60}]


@pytest.mark.parametrize("wrapper", [RecordingDevice, StealthDevice])
@pytest.mark.parametrize("mode", ["close", "context", "pool"])
def test_wrappers_close_inner_directly_and_through_pool(wrapper, mode):
    inner = TapDevice()
    inner.close = AsyncMock()
    device = wrapper(inner)

    async def run():
        if mode == "close":
            await device.close()
        elif mode == "context":
            async with device:
                pass
        else:
            await DevicePool([device]).close()

    asyncio.run(run())
    inner.close.assert_awaited_once()


class Writer(Extension):
    @device_action()
    async def write(self, device: Device, text: str) -> None:
        """Write a public label."""
        await device.input_text(text)

    @device_action(sensitive_parameters=frozenset({"text"}))
    async def private(self, device: Device, text: str) -> None:
        """Write a private label."""
        await self.write(text)


def test_sensitive_outer_action_records_only_redacted_reference():
    async def run():
        raw = MemoryDevice()
        recorder = RecordingDevice(raw)
        writer = await Writer(name="writer").bind(recorder)
        await writer.private("synthetic-label")
        assert raw.inputs == ["synthetic-label"]
        assert recorder.recorded_actions == [
            {
                "action_type": "extension_call",
                "name": "writer.private",
                "arguments": {"text": "<redacted>"},
            }
        ]

    asyncio.run(run())


def test_public_extension_write_replays_once():
    async def run():
        recorder = RecordingDevice(MemoryDevice())
        writer = await Writer(name="writer").bind(recorder)
        await writer.write("hello")
        target = MemoryDevice()
        await Writer(name="writer").bind(target)
        for action in recorder.recorded_actions:
            await MacroPlayer(target).replay_action(action)
        assert target.inputs == ["hello"]

    asyncio.run(run())


def test_type_secret_requires_credential_selector():
    class InputDevice(MemoryDevice):
        async def _input_secret(self, value: str) -> bool:
            return await self.input_text(value)

    async def run():
        target = InputDevice()
        await DictCredentials({"one": "demo-one", "two": "demo-two"}).bind(target)
        with pytest.raises(MacroError, match="package or name"):
            await MacroPlayer(target).replay_macro(
                {
                    "macro_schema_version": "2.0",
                    "actions": [{"action_type": "type_secret"}],
                }
            )
        assert target.inputs == []

    asyncio.run(run())


def test_replayed_wait_is_recorded():
    recorder = RecordingDevice(MemoryDevice())
    action = {"action_type": "wait", "duration": 0.001}
    asyncio.run(MacroPlayer(recorder).replay_action(action))
    assert recorder.recorded_actions == [action]


def test_stealth_word_pause_is_recorded(monkeypatch):
    recorder = RecordingDevice(MemoryDevice())
    monkeypatch.setattr(
        "mobilerun.core.devices.stealth.random.uniform", lambda *_: 0.001
    )
    assert asyncio.run(StealthDevice(recorder).input_text("a b"))
    assert recorder.recorded_actions == [
        {"action_type": "input_text", "text": "a", "clear": False},
        {"action_type": "input_text", "text": " ", "clear": False},
        {"action_type": "wait", "duration": 0.001},
        {"action_type": "input_text", "text": "b", "clear": False},
    ]


def test_macro_operator_replays_on_supplied_device(tmp_path):
    from mobilerun.macro import MacroOperator

    device = FakeDevice()
    result = asyncio.run(
        MacroOperator(delay_between_actions=0).execute(
            device, str(write_macro(tmp_path))
        )
    )
    assert result.success
    assert result.steps == 1
    assert device.taps == [(10, 20)]


def test_macro_operator_mismatch_carries_unexecuted_steps(tmp_path):
    from mobilerun.macro import MacroMismatch, MacroOperator

    device = FakeDevice()
    with pytest.raises(MacroMismatch) as caught:
        asyncio.run(
            MacroOperator(state_timeout=0).execute(
                device, str(write_macro(tmp_path, True))
            )
        )
    assert caught.value.step_index == 0
    assert caught.value.remaining_actions[0]["action_type"] == "tap"
    assert caught.value.current_state["phone_state"]["package"] == "other"
    assert device.taps == []


def test_mismatch_index_and_remaining_steps_exclude_completed_prefix(tmp_path):
    from mobilerun.macro import MacroMismatch, MacroOperator

    path = write_macro(tmp_path, True)
    data = json.loads(path.read_text())
    data["actions"].insert(0, {"action_type": "tap", "x": 1, "y": 2})
    data["actions"].append({"action_type": "input_text", "text": "never typed"})
    path.write_text(json.dumps(data))
    device = FakeDevice()
    with pytest.raises(MacroMismatch) as caught:
        asyncio.run(
            MacroOperator(state_timeout=0, delay_between_actions=0).execute(
                device, str(path)
            )
        )
    assert caught.value.step_index == 1
    assert len(caught.value.remaining_actions) == 2
    assert caught.value.remaining_actions[0]["x"] == 10
    assert device.taps == [(1, 2)]


def test_step_slice_preserves_absolute_mismatch_index(tmp_path):
    from mobilerun.macro import MacroMismatch, MacroOperator

    path = write_macro(tmp_path, True)
    data = json.loads(path.read_text())
    data["actions"].insert(0, {"action_type": "tap", "x": 1, "y": 2})
    path.write_text(json.dumps(data))
    device = FakeDevice()
    with pytest.raises(MacroMismatch) as caught:
        asyncio.run(
            MacroOperator(state_timeout=0, start_from_step=1).execute(device, str(path))
        )
    assert caught.value.step_index == 1
    assert device.taps == []


def test_zero_step_limit_executes_nothing(tmp_path):
    from mobilerun.macro import MacroOperator

    device = FakeDevice()
    result = asyncio.run(
        MacroOperator().execute(device, str(write_macro(tmp_path)), max_steps=0)
    )
    assert result.success
    assert result.steps == 0
    assert device.taps == []
