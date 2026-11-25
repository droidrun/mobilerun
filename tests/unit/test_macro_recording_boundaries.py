"""Regression tests for named action boundaries and recorded timing."""

import asyncio

import pytest

from mobilerun.core.credentials import DictCredentials
from mobilerun.core.devices import Device
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.devices.stealth import StealthDevice
from mobilerun.core.extensions import Extension, device_action
from mobilerun.macro.errors import MacroError
from mobilerun.macro.player import MacroPlayer
from tests.support.devices import MemoryDevice


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
