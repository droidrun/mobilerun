"""Resource actions work without a phone; device-dependent actions opt in."""

import asyncio

import pytest

from mobilerun.core.devices.base import Device
from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.extensions import Extension, action, device_action
from mobilerun.core.proxy import CloudFixedProxy, LocalProxy
from mobilerun.errors import (
    CredentialFieldNotFound,
    DeviceOperationNotSupported,
    ExtensionActionError,
    ExtensionNotBound,
    MobilerunError,
)
from tests.support.devices import MemoryDevice


class Resource(Extension):
    @action()
    async def read(self) -> str:
        """Read resource metadata without a device."""
        return "available"

    @device_action
    async def device_name(self, device: Device) -> str:
        """Read a device identifier when explicitly bound."""
        self.received = device
        return device.identifier or "unknown"

    @action()
    async def fail(self) -> None:
        """Fail without exposing private backend details."""
        raise RuntimeError("private backend value")


def test_unbound_direct_and_catalog_actions_work_and_wrap_errors():
    async def scenario():
        resource = Resource(name="resource")
        assert await resource.read() == "available"
        assert await resource.actions.call("resource.read", {}) == "available"
        for call in (resource.fail, lambda: resource.actions.call("resource.fail", {})):
            with pytest.raises(ExtensionActionError) as raised:
                await call()
            assert "private backend value" not in str(raised.value)

    asyncio.run(scenario())


def test_device_action_rejects_before_body_and_injects_facade():
    async def scenario():
        resource = Resource(name="vault")
        for call in (
            resource.device_name,
            lambda: resource.actions.call("vault.device_name", {}),
        ):
            with pytest.raises(
                ExtensionNotBound,
                match=r"needs a bound device; await vault.bind\(device\) first",
            ):
                await call()
        assert not hasattr(resource, "received")
        phone = DeviceWrapper(MemoryDevice())
        await resource.bind(phone)
        assert await resource.device_name() == "memory"
        assert resource.received is phone
        assert await resource.actions.call("vault.device_name", {}) == "memory"
        assert resource.received is phone
        await phone.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("uncertain", [False, True])
def test_bound_uncertain_and_draining_checks_still_apply(uncertain):
    async def scenario():
        resource = Resource()
        phone = MemoryDevice()
        await resource.bind(phone)
        if uncertain:
            resource._binding.uncertain = True
        else:
            phone._extension_state.closing = True
        for call in (
            resource.read,
            lambda: resource.actions.call("extension.read", {}),
        ):
            with pytest.raises(
                ExtensionNotBound, match="uncertain" if uncertain else "closing"
            ):
                await call()
        await phone.close()

    asyncio.run(scenario())


def test_unbound_actions_keep_serialization_and_cross_extension_guard():
    active = 0
    peak = 0

    class Counter(Extension):
        @action()
        async def read(self) -> int:
            """Check action serialization without a device."""
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return 1

        @action()
        async def cross(self) -> int:
            """Attempt a cross-extension unbound action."""
            return await other.read()

    async def scenario():
        resource = Counter()
        await asyncio.gather(
            *(resource.read() for _ in range(5)),
            *(resource.actions.call("extension.read", {}) for _ in range(5)),
        )
        assert peak == 1
        with pytest.raises(ExtensionActionError, match="Cross-extension"):
            await resource.cross()

    other = Counter(name="other")
    asyncio.run(asyncio.wait_for(scenario(), timeout=1))


def test_local_proxy_status_explicitly_requires_device():
    proxy = LocalProxy("localhost", 8080)
    with pytest.raises(
        ExtensionNotBound,
        match=r"needs a bound device; await proxy.bind\(device\) first",
    ):
        asyncio.run(proxy.status())


def test_cloud_proxy_status_works_unbound():
    assert not asyncio.run(CloudFixedProxy("known").status()).connected


def test_device_action_metadata_schema_and_signature():
    import inspect

    class Typed(Extension):
        @device_action(public=False, sensitive_parameters=frozenset({"value"}))
        async def write(self, device: Device, value: str) -> str:
            """Return a typed value with a hidden injected device."""
            return value

    resource = Typed()
    entry = resource.actions.get("extension.write")
    assert entry.requires_device
    assert not entry.public
    assert set(entry.input_schema["properties"]) == {"value"}
    assert list(inspect.signature(resource.write).parameters) == ["value"]
    assert not Resource().actions.get("extension.read").requires_device


def test_inherited_device_action_override_keeps_injection():
    class Override(Resource):
        async def device_name(self, device: Device) -> str:
            """Override a device action without repeating its decorator."""
            self.received = device
            return "override"

    async def scenario():
        resource = Override()
        assert resource.actions.get("extension.device_name").requires_device
        with pytest.raises(ExtensionNotBound):
            await resource.actions.call("extension.device_name", {})
        phone = MemoryDevice()
        await resource.bind(phone)
        assert await resource.actions.call("extension.device_name", {}) == "override"
        assert resource.received is phone
        await phone.close()

    asyncio.run(scenario())


def test_nested_device_actions_inject_facade_and_reject_unbound_before_body():
    class Nested(Extension):
        ran = False

        @device_action()
        async def inner(self, device: Device, value: str) -> str:
            """Use the injected facade in a nested call."""
            self.ran = True
            assert device is self.device
            return value

        @action
        async def outer(self, value: str) -> str:
            """Delegate to a device-dependent action on the same extension."""
            return await self.inner(value)

    async def scenario():
        resource = Nested()
        with pytest.raises(
            ExtensionNotBound, match="extension.inner.*needs a bound device"
        ):
            await resource.outer("value")
        assert not resource.ran
        facade = DeviceWrapper(MemoryDevice())
        await resource.bind(facade)
        assert await resource.outer("value") == "value"
        assert resource.ran
        await facade.close()

    asyncio.run(asyncio.wait_for(scenario(), timeout=1))


def test_device_action_rejects_missing_or_wrong_device_annotation():
    with pytest.raises(TypeError, match="device: Device"):

        @device_action
        async def missing(self) -> None:
            """Missing the injected parameter."""

    with pytest.raises(TypeError, match="device: Device"):

        @device_action
        async def wrong(self, device: str) -> None:
            """Incorrectly annotated injected parameter."""


@pytest.mark.parametrize(
    "error",
    [
        CredentialFieldNotFound("Credential field not found"),
        DeviceOperationNotSupported("Operation unavailable"),
    ],
)
def test_value_safe_action_errors_preserve_type_and_identity(error):
    class Failing(Extension):
        @action
        async def run(self) -> None:
            """Raise a public value-safe typed error."""
            raise error

    async def scenario():
        resource = Failing()
        for invoke in (
            resource.run,
            lambda: resource.actions.call("extension.run", {}),
        ):
            with pytest.raises(type(error)) as raised:
                await invoke()
            assert raised.value is error

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "error",
    [RuntimeError("CANARY-PRIVATE-VALUE"), MobilerunError("CANARY-PRIVATE-VALUE")],
)
def test_unsafe_action_errors_remain_wrapped_without_values(error):
    class Failing(Extension):
        @action
        async def run(self) -> None:
            """Raise an unsafe backend failure."""
            raise error

    async def scenario():
        resource = Failing()
        for invoke in (
            resource.run,
            lambda: resource.actions.call("extension.run", {}),
        ):
            with pytest.raises(ExtensionActionError) as raised:
                await invoke()
            assert "CANARY-PRIVATE-VALUE" not in str(raised.value)
            assert raised.value.__suppress_context__

    asyncio.run(scenario())
