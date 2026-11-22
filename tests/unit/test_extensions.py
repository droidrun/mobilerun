"""Public extension contracts, binding invariants and schemas."""

import asyncio
from enum import Enum

import pytest


class Choice(str, Enum):
    ONE = "one"


def test_public_contract_kit():
    from mobilerun.core.extensions import Extension, action
    from mobilerun.models.context import ContextItem
    from tests.support.devices import MemoryDevice
    from tests.support.extensions import check_extension_contract

    class Authenticator(Extension):
        default_name = "auth"
        exclusive = "authenticator"

        @action()
        async def code(self, label: str = "work") -> str:
            """Return a deterministic test OTP, never a seed."""
            return "123456"

        async def context(self):
            return (ContextItem(self.name, "authenticator", {"method": "totp"}),)

    async def device():
        return MemoryDevice()

    asyncio.run(check_extension_contract(lambda name: Authenticator(name=name), device))


def test_schema_and_call_preserve_enums_and_null_defaults():
    from mobilerun.core.extensions import Extension, action
    from tests.support.devices import MemoryDevice

    class Example(Extension):
        default_name = "example"

        @action()
        async def read(self, choice: Choice, optional: str | None = None) -> list[str]:
            """Return selected values."""
            return [choice.value, optional or "default"]

    async def scenario():
        device = MemoryDevice()
        ext = Example()
        await ext.bind(device)
        entry = device.actions.get("example.read")
        assert "$defs" in entry.input_schema
        assert entry.input_schema["required"] == ["choice"]
        assert await device.actions.call("example.read", {"choice": "one"}) == [
            "one",
            "default",
        ]
        await device.close()

    asyncio.run(scenario())


def test_action_rejects_sync_and_missing_docs():
    from mobilerun.core.extensions import action

    with pytest.raises(TypeError, match="async"):
        action()(lambda: None)

    async def undocumented() -> None:
        pass

    with pytest.raises(TypeError, match="docstring"):
        action()(undocumented)


def test_compatibility_precedes_force_and_reserved_names_fail():
    from mobilerun.core.extensions import Extension
    from mobilerun.errors import ExtensionConflict, ExtensionNotSupported
    from tests.support.devices import MemoryDevice

    class OnlyCloud(Extension):
        default_name = "cloud_ext"
        exclusive = "kind"

        async def check_bind(self, device):
            conflicts = await super().check_bind(device)
            if device.cloud_id is None:
                raise ExtensionNotSupported("requires cloud")
            return conflicts

    async def scenario():
        device = MemoryDevice()
        old = Extension(name="old")
        await old.bind(device)
        with pytest.raises(ExtensionNotSupported):
            await OnlyCloud().bind(device, force=True)
        with pytest.raises(ExtensionConflict, match="reserved"):
            await Extension(name="close").bind(device)
        assert tuple(device.extensions) == ("old",)
        await device.close()

    asyncio.run(scenario())


def test_named_actions_record_through_outer_wrapper():
    from mobilerun.core.devices.recording import RecordingDevice
    from mobilerun.core.extensions import Extension, action
    from tests.support.devices import MemoryDevice

    class Example(Extension):
        @action()
        async def echo(self, value: str) -> str:
            """Return a public value."""
            return value

    async def scenario():
        raw = MemoryDevice(supported_operations=frozenset({"input_text"}))
        recording = RecordingDevice(raw)
        await recording.actions.call("device.input_text", {"text": "hello"})
        ext = Example(name="example")
        await ext.bind(raw)
        await ext.echo("world")
        assert recording.recorded_actions == [
            {"action_type": "input_text", "text": "hello", "clear": False},
            {
                "action_type": "extension_call",
                "name": "example.echo",
                "arguments": {"value": "world"},
            },
        ]
        await recording.close()

    asyncio.run(scenario())


def test_undecorated_action_override_keeps_recording():
    from mobilerun.core.devices.recording import RecordingDevice
    from mobilerun.core.extensions import Extension, action
    from tests.support.devices import MemoryDevice

    class Parent(Extension):
        @action()
        async def value(self) -> int:
            """Return the current value."""
            return 1

    class Child(Parent):
        async def value(self) -> int:
            """Return the overridden value."""
            return 2

    async def scenario():
        device = RecordingDevice(MemoryDevice())
        child = Child(name="child")
        await child.bind(device)
        assert await child.value() == 2
        assert device.recorded_actions[0]["name"] == "child.value"
        await device.close()

    asyncio.run(scenario())


def test_close_cannot_race_a_queued_bind():
    from mobilerun.core.extensions import Extension
    from mobilerun.errors import ExtensionConflict
    from tests.support.devices import MemoryDevice

    async def scenario():
        extension = Extension()
        phone = MemoryDevice()
        await extension._lifecycle_lock.acquire()
        closing = asyncio.create_task(extension.close())
        await asyncio.sleep(0)
        binding = asyncio.create_task(extension.bind(phone))
        await asyncio.sleep(0)
        extension._lifecycle_lock.release()
        await closing
        try:
            await binding
        except ExtensionConflict:
            pass
        assert extension.device is None
        assert not phone.extensions
        await phone.close()

    asyncio.run(scenario())


def test_failed_force_finishes_rollback_before_old_close():
    from mobilerun.core.extensions import Extension
    from tests.support.devices import MemoryDevice

    async def scenario():
        started, finish = asyncio.Event(), asyncio.Event()
        events = []

        class Old(Extension):
            exclusive = "resource"

            async def on_bind(self, device):
                events.append("bind")

            async def on_unbind(self, device):
                events.append("unbind")

            async def on_close(self):
                events.append("close")

        class Incoming(Extension):
            exclusive = "resource"

            async def on_bind(self, device):
                started.set()
                await finish.wait()
                raise RuntimeError("incoming failed")

        phone = MemoryDevice()
        old, incoming = Old(name="old"), Incoming(name="incoming")
        await old.bind(phone)
        switching = asyncio.create_task(incoming.bind(phone, force=True))
        await started.wait()
        closing = asyncio.create_task(old.close())
        await asyncio.sleep(0)
        finish.set()
        with pytest.raises(RuntimeError, match="incoming failed"):
            await switching
        await closing
        assert events == ["bind", "unbind", "bind", "unbind", "close"]
        assert old.device is None
        assert not phone.extensions
        await phone.close()

    asyncio.run(scenario())
