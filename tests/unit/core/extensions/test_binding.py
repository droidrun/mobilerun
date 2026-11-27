"""Extension binding, action dispatch, and optional resource contracts."""

import asyncio
import json
from enum import Enum
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.credentials import (
    CloudCredentials,
    DictCredentials,
    EnvCredentials,
    FileCredentials,
)
from mobilerun.core.devices import (
    AndroidDevice,
    CloudDevice,
    IOSDevice,
    RecordingDevice,
    StealthDevice,
)
from mobilerun.core.devices.base import Device
from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.extensions import Extension, action, device_action
from mobilerun.core.mailbox import CloudMailbox
from mobilerun.core.proxy import CloudFixedProxy, CloudRotatingProxy, LocalProxy
from mobilerun.errors import (
    CredentialFieldNotFound,
    DeviceOperationNotSupported,
    ExtensionActionError,
    ExtensionConflict,
    ExtensionNotBound,
    ExtensionNotSupported,
    MobilerunError,
    ProxyNotActive,
)
from mobilerun.models.context import ContextItem
from tests.support.devices import MemoryDevice
from tests.support.extensions import check_extension_contract


class Choice(str, Enum):
    ONE = "one"


def test_public_contract_kit():
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
    with pytest.raises(TypeError, match="async"):
        action()(lambda: None)

    async def undocumented() -> None:
        pass

    with pytest.raises(TypeError, match="docstring"):
        action()(undocumented)


def test_compatibility_precedes_force_and_reserved_names_fail():
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


def test_named_actions_record_through_outer_wrapper(tmp_path):
    class Example(Extension):
        @action()
        async def echo(self, value: str) -> str:
            """Return a public value."""
            return value

    async def scenario():
        raw = MemoryDevice(supported_operations=frozenset({"input_text"}))
        recording = RecordingDevice(raw, runs_dir=tmp_path, capture="actions")
        await recording.actions.call("device.input_text", {"text": "hello"})
        ext = Example(name="example")
        await ext.bind(raw)
        await recording.example.echo("world")
        lines = (recording.run_dir / "steps.jsonl").read_text().splitlines()
        assert [(row["action"], row["args"]) for row in map(json.loads, lines)] == [
            (
                "input_text",
                {"text": "hello", "clear": False, "stealth": False, "wpm": 0},
            ),
            (
                "extension_call",
                {"name": "example.echo", "arguments": {"value": "world"}},
            ),
        ]
        await recording.close()

    asyncio.run(scenario())


def test_undecorated_action_override_keeps_recording(tmp_path):
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
        device = RecordingDevice(MemoryDevice(), runs_dir=tmp_path, capture="actions")
        child = Child(name="child")
        await child.bind(device)
        assert await child.value() == 2
        row = json.loads((device.run_dir / "steps.jsonl").read_text())
        assert row["args"]["name"] == "child.value"
        await device.close()

    asyncio.run(scenario())


def test_observer_records_outer_dispatch_outcomes_without_primitive_duplicates(
    tmp_path,
):
    class Example(Extension):
        @device_action()
        async def write(self, device: Device, mode: str) -> bool:
            """Write the requested value and return a status."""
            await device.input_text(mode)
            if mode == "raise":
                raise ValueError("backend details")
            return mode == "ok"

    async def scenario():
        raw = MemoryDevice()
        recording = RecordingDevice(raw, runs_dir=tmp_path, capture="actions")
        ext = await Example().bind(recording)
        with pytest.raises(ExtensionActionError):
            await recording.actions.call("extension.write", {"mode": 7})
        assert recording.run_dir is None
        assert await ext.write("ok") is True
        assert await ext.write("false") is False
        with pytest.raises(ExtensionActionError):
            await ext.write("raise")
        lines = (recording.run_dir / "steps.jsonl").read_text().splitlines()
        assert [(r["action"], r["ok"], r["error"]) for r in map(json.loads, lines)] == [
            ("extension_call", True, None),
            ("extension_call", False, "returned_false"),
            ("extension_call", False, "ValueError"),
        ]
        assert raw.inputs == ["ok", "false", "raise"]

    asyncio.run(scenario())


def test_nested_sensitive_action_redacts_public_aliases(tmp_path):
    class Example(Extension):
        @action()
        async def outer(self, public_alias: str) -> None:
            """Pass a value to the nested action."""
            await self.inner(public_alias)

        @action(sensitive_parameters=frozenset({"secret"}))
        async def inner(self, secret: str) -> None:
            """Accept a sensitive value."""
            pass

    async def scenario():
        recording = RecordingDevice(
            MemoryDevice(), runs_dir=tmp_path, capture="actions"
        )
        extension = await Example().bind(recording)
        await extension.outer("private-canary")
        evidence = (recording.run_dir / "steps.jsonl").read_text()
        assert "private-canary" not in evidence
        row = json.loads(evidence)
        assert row["args"]["arguments"] == {"public_alias": "***"}

    asyncio.run(scenario())


def test_stealth_catalog_vetoes_extension_bound_to_raw_sibling(tmp_path):
    class Example(Extension):
        @action()
        async def ready(self) -> bool:
            """Return readiness."""
            return True

    async def scenario():
        raw = MemoryDevice()
        recording = RecordingDevice(raw, runs_dir=tmp_path, capture="actions")
        await Example().bind(raw)
        assert (
            await StealthDevice(recording).actions.call("extension.ready", {}) is True
        )
        assert recording.run_dir is None

    asyncio.run(scenario())


def test_only_invoking_recorder_observes_direct_and_catalog_extension_calls(tmp_path):
    class Example(Extension):
        @action()
        async def ready(self) -> bool:
            """Return readiness."""
            return True

    async def scenario():
        raw = MemoryDevice()
        first = RecordingDevice(raw, runs_dir=tmp_path, capture="actions")
        second = RecordingDevice(raw, runs_dir=tmp_path, capture="actions")
        await Example().bind(first)
        assert await StealthDevice(first).extension.ready()
        assert await StealthDevice(first).extensions["extension"].ready()
        assert await raw.extension.ready()
        assert first.run_dir is None and second.run_dir is None
        assert await second.extension.ready()
        assert first.step_index == 0 and second.step_index == 1
        assert await first.actions.call("extension.ready", {})
        assert first.step_index == second.step_index == 1

    asyncio.run(scenario())


def test_dispatch_guard_blocks_nested_commerce_even_without_recording():
    from mobilerun.core.extensions import dispatch_guard

    calls = []

    async def retry(owner, policy):
        calls.append("paid")

    class Example(Extension):
        role = "phone_number"

        @action()
        async def outer(self) -> None:
            """Call the nested method."""
            await self.buy()

        @action(billing_retry=retry)
        async def buy(self) -> None:
            """Simulate a paid dispatch."""
            calls.append("paid")

    def deny(info):
        if info.supports_billing_retry:
            assert (info.extension_role, info.extension_name, info.action_name) == (
                "phone_number",
                "example",
                "buy",
            )
            raise ExtensionActionError("commerce blocked")

    async def scenario():
        example = Example(name="example")
        with dispatch_guard(deny), dispatch_guard(lambda info: None):
            with pytest.raises(ExtensionActionError, match="commerce blocked"):
                await example.outer()
        assert calls == []
        await example.outer()
        assert calls == ["paid"]

    asyncio.run(scenario())


def test_close_cannot_race_a_queued_bind():
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


@pytest.mark.parametrize("kind", ["dict", "file", "env", "cloud"])
def test_every_provider_binds_and_unbinds_in_order(kind, tmp_path, monkeypatch):
    path = tmp_path / "credentials.yaml"
    path.write_text("login: example-only\n")
    path.chmod(0o600)
    monkeypatch.setenv("MOBILERUN_SECRET_LOGIN", "example-only")
    provider = {
        "dict": lambda: DictCredentials({}, name="second"),
        "file": lambda: FileCredentials(path, name="second"),
        "env": lambda: EnvCredentials(name="second"),
        "cloud": lambda: CloudCredentials(api_key="fake", name="second"),
    }[kind]()
    phone = AndroidDevice("local")
    first = DictCredentials({"login": "first"})

    async def exercise():
        await first.bind(phone)
        wrapped = StealthDevice(RecordingDevice(phone))
        await provider.bind(wrapped)
        await provider.bind(phone)
        assert list(phone.extensions) == list(wrapped.extensions) == ["vault", "second"]
        assert phone.extensions["vault"] is first
        assert wrapped.extensions["second"] is provider
        await first.unbind()
        assert list(phone.extensions) == ["second"]
        await provider.unbind()
        await provider.unbind()
        assert not phone.extensions
        if isinstance(provider, CloudCredentials):
            await provider.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AndroidDevice("local", credentials=[]),
        lambda: IOSDevice("http://local", credentials=[]),
        lambda: CloudDevice(api_key="fake", credentials=[]),
    ],
)
def test_device_credentials_constructor_is_removed(factory):
    with pytest.raises(
        TypeError, match="provider.bind|unexpected keyword argument 'credentials'"
    ):
        factory()


def test_mailbox_binding_is_local_and_supports_local_typing():
    phone = AndroidDevice("local")
    mailbox = CloudMailbox("mailbox", api_key="fake")
    mailbox.wait_for_code = AsyncMock(return_value="123456")

    async def exercise():
        await mailbox.bind(RecordingDevice(phone))
        assert phone.extensions[mailbox.name].device is mailbox.device
        assert await phone.mailbox.wait_for_code() == "123456"
        await mailbox.unbind()
        with pytest.raises(DeviceOperationNotSupported, match="No email extension"):
            _ = phone.mailbox

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "connected,restarted", [(True, False), (False, False), (True, True)]
)
def test_phone_proxy_switch_checks_status_and_waits_after_restart(
    monkeypatch, connected, restarted
):
    from mobilerun.clients.cloud import proxies

    phone = CloudDevice("phone", api_key="fake", billing="paid")
    connect = AsyncMock()
    monkeypatch.setattr(proxies, "connect_device", connect)
    phone.info = AsyncMock(
        return_value=NS(state="restarting" if restarted else "ready")
    )
    phone.wait_ready = AsyncMock()
    from mobilerun.core.proxy import ProxyStatus

    phone._proxy_status = AsyncMock(return_value=ProxyStatus(connected=connected))
    proxy = CloudRotatingProxy(country="DE")

    async def exercise():
        if connected:
            await proxy.bind(phone, force=True)
            assert (await proxy.status()).connected
            assert phone.extensions["proxy"] is proxy
        else:
            with pytest.raises(ProxyNotActive):
                await proxy.bind(phone, force=True)
            assert not phone.extensions

    asyncio.run(exercise())
    assert phone.wait_ready.await_count == int(restarted)
    connect.assert_awaited_once()
    assert proxy.exclusive == "proxy"


def test_default_proxy_uses_smart_ip_setting(monkeypatch):
    monkeypatch.setenv("MOBILERUN_CLOUD_PROXY_SMART_IP", "false")
    assert CloudRotatingProxy._from_settings().smart_ip is False
