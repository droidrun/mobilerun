"""Extension lifecycle, shared device operations, and sensitive action contracts."""

import asyncio
import copy
import json
import logging
import pickle
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Literal, cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, Field

from mobilerun.core.devices.base import Device
from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.devices.pool import DevicePool
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.extensions import Extension, action, device_action
from mobilerun.core.extensions.registry import Binding
from mobilerun.core.phone_number import CloudPhoneNumber
from mobilerun.core.proxy import CloudFixedProxy, ExistingCloudProxy, LocalProxy
from mobilerun.errors import (
    DeviceOperationNotSupported,
    ExtensionActionError,
    ExtensionClosed,
    ExtensionConflict,
    ExtensionNotBound,
)
from mobilerun.models import CloudCredentialReference, CloudTaskInputs
from mobilerun.models.proxy import ProxyStatus
from tests.support.devices import MemoryDevice
from tests.support.extensions import check_extension_contract
from tests.support.proxy import ProxyMemory


def test_reciprocal_cross_extension_actions_fail_without_deadlock():
    async def scenario():
        ready = asyncio.Barrier(2)

        class Peer(Extension):
            @action()
            async def inner(self) -> None:
                """Return without doing work."""

            @action()
            async def outer(self) -> None:
                """Attempt a reciprocal nested action."""
                await ready.wait()
                await peers[1 if self is peers[0] else 0].inner()

        phone = MemoryDevice()
        peers = [Peer(name="first"), Peer(name="second")]
        for peer in peers:
            await peer.bind(phone)
        results = await asyncio.gather(
            *(peer.outer() for peer in peers), return_exceptions=True
        )
        assert all(
            isinstance(result, ExtensionActionError)
            and "Cross-extension" in str(result)
            for result in results
        )
        await phone.close()

    asyncio.run(asyncio.wait_for(scenario(), timeout=1))


def test_unbind_waiting_for_cross_extension_call_finishes():
    async def scenario():
        entered, proceed = asyncio.Event(), asyncio.Event()

        class Peer(Extension):
            @action()
            async def inner(self) -> None:
                """Return without doing work."""

            @action()
            async def outer(self) -> None:
                """Try another extension while unbind is waiting."""
                entered.set()
                await proceed.wait()
                await second.inner()

        phone = MemoryDevice()
        first, second = Peer(name="first"), Peer(name="second")
        await first.bind(phone)
        await second.bind(phone)
        running = asyncio.create_task(first.outer())
        await entered.wait()
        unbinding = asyncio.create_task(first.unbind())
        await asyncio.sleep(0)
        proceed.set()
        with pytest.raises(ExtensionActionError, match="Cross-extension"):
            await running
        await unbinding
        assert first.device is None
        await phone.close()

    asyncio.run(asyncio.wait_for(scenario(), timeout=1))


@pytest.mark.parametrize("detached", [False, True])
def test_close_waits_for_unbound_action_and_rejects_new_actions(detached):
    async def scenario():
        entered, proceed = asyncio.Event(), asyncio.Event()

        class Resource(Extension):
            released = False

            @action
            async def inner(self) -> bool:
                """Finish same-extension work while shutdown waits."""
                assert not self.released
                return True

            @action
            async def read(self) -> bool:
                """Hold an unbound resource until the caller permits completion."""
                entered.set()
                await proceed.wait()
                assert not self.released
                return await self.inner()

            async def on_close(self) -> None:
                self.released = True

        resource = Resource()
        phone = MemoryDevice()
        if detached:
            await resource.bind(phone)
            await resource.unbind()
        running = asyncio.create_task(resource.read())
        await entered.wait()
        # This call queues behind the running body; shutdown must reject it too.
        queued = asyncio.create_task(resource.inner())
        await asyncio.sleep(0)
        closing = asyncio.create_task(resource.close())
        await asyncio.sleep(0)
        assert not closing.done()
        assert not resource.released
        with pytest.raises(ExtensionClosed):
            await resource.inner()
        proceed.set()
        assert await running
        with pytest.raises(ExtensionClosed):
            await queued
        await closing
        assert resource.released
        for invoke in (
            resource.inner,
            lambda: resource.actions.call("extension.inner", {}),
        ):
            with pytest.raises(ExtensionClosed) as raised:
                await invoke()
            assert raised.value.value_safe
        await resource.close()
        await phone.close()

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


@pytest.mark.parametrize("mutating", [False, True])
def test_cancelled_close_only_stays_closed_after_mutation_started(mutating):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        class Resource(Extension):
            @action
            async def read(self) -> bool:
                """Safely read the resource."""
                return True

            async def on_close(self) -> None:
                entered.set()
                await release.wait()

        resource = Resource()
        if not mutating:
            await resource._action_lock.acquire()
        closing = asyncio.create_task(resource.close())
        if mutating:
            await entered.wait()
        else:
            await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        if not mutating:
            resource._action_lock.release()
            assert await resource.read()
        else:
            with pytest.raises(ExtensionClosed):
                await resource.read()
        release.set()
        await resource.close()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_refused_cloud_proxy_close_does_not_poison_the_handle():
    async def run():
        phone = ProxyMemory(True)
        proxy = CloudFixedProxy("old")
        await proxy.bind(phone)
        with pytest.raises(ExtensionConflict):
            await proxy.close()
        assert proxy.device is phone
        assert (await proxy.status()).connected
        assert not proxy._closing
        await phone.close()

    asyncio.run(run())


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_failed_local_detach_reconciles_actual_remote_state(force, cancel):
    async def scenario():
        phone = ProxyMemory(False)
        old = LocalProxy("host", 8080)
        incoming = LocalProxy("new", 8081)
        await old.bind(phone)
        detached = asyncio.Event()
        original = RuntimeError("detached then failed")

        async def disconnect():
            phone.target.current = ProxyStatus(False)
            detached.set()
            if cancel:
                await asyncio.Event().wait()
            raise original

        phone.target.disconnect = disconnect
        task = asyncio.create_task(
            incoming.bind(phone, force=True) if force else old.unbind()
        )
        await detached.wait()
        if cancel:
            task.cancel()
        with pytest.raises(
            asyncio.CancelledError if cancel else RuntimeError
        ) as raised:
            await task
        if not cancel:
            assert raised.value is original
        assert not phone.extensions
        assert old.device is None
        assert incoming.device is None
        await phone.close()

    asyncio.run(asyncio.wait_for(scenario(), timeout=1))


@pytest.mark.parametrize("force", [False, True])
def test_failed_detach_with_unreadable_status_is_uncertain(force):
    async def scenario():
        phone = ProxyMemory(False)
        old = LocalProxy("host", 8080)
        incoming = LocalProxy("new", 8081)
        await old.bind(phone)
        original = RuntimeError("detach failed")
        phone.target.disconnect = AsyncMock(side_effect=original)
        phone.target.status = AsyncMock(side_effect=RuntimeError("status unreadable"))
        with pytest.raises(RuntimeError) as raised:
            if force:
                await incoming.bind(phone, force=True)
            else:
                await old.unbind()
        assert raised.value is original
        assert phone.proxy is old
        assert old._binding.uncertain
        assert any("uncertain" in note for note in original.__notes__)
        with pytest.raises(ExtensionNotBound, match="uncertain"):
            await old.status()
        assert incoming.device is None
        phone.target.disconnect = AsyncMock()
        await phone.close()

    asyncio.run(scenario())


def test_successful_rollback_with_different_unresolvable_reference_does_not_preserve_handle():
    async def run():
        phone = ProxyMemory(True)
        old = CloudFixedProxy("old")
        await old.bind(phone)
        original = RuntimeError("switch failed")

        async def connect(name):
            phone.target.current = ProxyStatus(True, "different-reference")
            if name == "new":
                raise original
            return ProxyStatus(True, "mrc:old")

        phone.target.connect.side_effect = connect
        phone.target._resolve_reference = AsyncMock(return_value=None)
        with pytest.raises(RuntimeError) as raised:
            await CloudFixedProxy("new").bind(phone, force=True)
        assert raised.value is original
        assert isinstance(phone.proxy, ExistingCloudProxy)
        assert old.device is None
        await phone.close()

    asyncio.run(run())


@pytest.mark.parametrize("connected", [False, True])
def test_number_readback_uses_extension_owned_identity_and_update_hooks(connected):
    async def run():
        phone = MemoryDevice(
            cloud_id="device", supported_operations=frozenset({"cloud_telephony"})
        )

        class ThirdPartyNumber(Extension):
            role = "phone_number"
            exclusive = "phone_number"
            observed = "not-called"

            def _matches_remote(self, observed: object) -> bool:
                return getattr(observed, "number_id", None) == "third-party"

            def _observe_remote_binding(
                self, device: Device, observed: object | None
            ) -> None:
                assert device is phone
                self.observed = observed

        original = ThirdPartyNumber(name="sms")
        await original.bind(phone, close_on_device_close=True)
        binding = original._binding
        remote = (
            SimpleNamespace(number_id="third-party", phone_number="+123")
            if connected
            else None
        )
        with patch(
            "mobilerun.core.phone_number.cloud.devices.number",
            AsyncMock(return_value=SimpleNamespace(binding=remote)),
        ):
            await CloudPhoneNumber("incoming")._reconcile_bind_failure(
                phone, RuntimeError("response lost")
            )
        if connected:
            assert original.observed.number_id == remote.number_id
            assert original.observed.phone_number == remote.phone_number
            assert phone.number is original
            assert original._binding is binding
            assert binding.owned
        else:
            assert not phone.extensions
            assert original.device is None
        await phone.close()

    asyncio.run(run())


def test_contract_checker_detects_orphan_inserted_after_failed_bind():
    class Broken(Extension):
        async def bind(self, device, **kwargs):
            try:
                return await super().bind(device, **kwargs)
            except Exception:
                orphan = Extension(name="orphan")
                device._extension_state.bindings["orphan"] = Binding(
                    orphan, device, False
                )
                raise

    async def device():
        return MemoryDevice()

    with pytest.raises(AssertionError, match="C:.*registry"):
        asyncio.run(check_extension_contract(lambda name: Broken(name=name), device))


@pytest.mark.parametrize("secret_view", ["root", "wrapper", "pool"])
@pytest.mark.parametrize("other_view", ["root", "wrapper", "pool", "catalog"])
@pytest.mark.parametrize("operation", ["tap", "input_text"])
def test_focus_precheck_and_secret_typing_are_atomic_across_views(
    secret_view, other_view, operation
):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        class FocusDevice(MemoryDevice):
            focus = "password"

            def __init__(self):
                super().__init__(supported_operations=frozenset({"tap", "input_text"}))
                self.timeline = []

            async def _input_text_raw(
                self,
                text: str,
                clear: bool = False,
                stealth: bool = False,
                wpm: int = 0,
            ) -> bool:
                if text == "secret-canary":
                    entered.set()
                    await release.wait()
                self.timeline.append((text, self.focus))
                return True

            async def tap(self, x: int, y: int) -> None:
                self.timeline.append(("tap", self.focus))
                self.focus = "chat"

        class Vault(Extension):
            @device_action
            async def type_field(self, device: Device) -> bool:
                """Type a non-secret test canary through the bound facade."""
                return await device._input_secret("secret-canary")

        root = FocusDevice()
        wrapper = DeviceWrapper(RecordingDevice(root))
        for view in (root, wrapper):
            view.state.refresh = AsyncMock(
                return_value=SimpleNamespace(
                    phone_state={
                        "focusedElement": {"id": "password"},
                        "isEditable": True,
                    }
                )
            )
        pool = DevicePool([wrapper])
        vault = Vault(name="vault")
        await vault.bind(wrapper)
        if secret_view == "pool":
            secret = asyncio.create_task(pool.actions.call("vault.type_field", {}))
        else:
            secret = asyncio.create_task(
                (root if secret_view == "root" else wrapper)._input_secret(
                    "secret-canary"
                )
            )
        await entered.wait()
        arguments = {"x": 1, "y": 1} if operation == "tap" else {"text": "plain"}
        if other_view == "catalog":
            other = asyncio.create_task(
                root.actions.call(f"device.{operation}", arguments)
            )
        else:
            view = {"root": root, "wrapper": wrapper, "pool": pool}[other_view]
            other = asyncio.create_task(getattr(view, operation)(**arguments))
        for _ in range(3):
            await asyncio.sleep(0)
        assert not other.done()
        assert root.focus == "password"
        release.set()
        await secret
        await other
        assert root.timeline[0] == ("secret-canary", "password")
        assert root.timeline[1][0] == ("tap" if operation == "tap" else "plain")
        await root.close()
        await vault.close()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("swipe", (0, 0, 1, 1)),
        ("drag", (0, 0, 1, 1)),
        ("press_button", ("home",)),
        ("press_key_code", (1,)),
        ("start_app", ("app",)),
        ("stop_app", ("app",)),
        ("open_url", ("https://example.com",)),
        ("tap_element", (1,)),
    ],
)
def test_other_focus_changing_operations_share_the_root_lock(name, arguments):
    async def run():
        root = MemoryDevice()
        wrapper = DeviceWrapper(root)
        async with root._input_operations.hold():
            queued = asyncio.create_task(getattr(wrapper, name)(*arguments))
            await asyncio.sleep(0)
            assert not queued.done()
        # The in-memory backend may not implement the operation. It must only
        # reach its body after the root input lock has been released.
        await asyncio.gather(queued, return_exceptions=True)

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_child_tasks_do_not_inherit_input_lock_ownership_and_cancel_releases_it():
    async def run():
        root = MemoryDevice()
        async with root._input_operations.hold():
            # Same-task nested delegation is reentrant.
            assert await DeviceWrapper(root).input_text("first")
            child = asyncio.create_task(root.input_text("cancelled"))
            await asyncio.sleep(0)
            assert not child.done()
            child.cancel()
            with pytest.raises(asyncio.CancelledError):
                await child
        assert await root.input_text("last")
        assert root.inputs == ["first", "last"]

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_inherited_mixin_operations_and_independent_roots_use_the_correct_lock():
    class NavigationMixin:
        async def open_url(self, url: str) -> None:
            self.opened = url

    class MixedDevice(NavigationMixin, MemoryDevice):
        pass

    async def run():
        root = MixedDevice()
        independent = MemoryDevice()
        async with root._input_operations.hold():
            pending = asyncio.create_task(
                DeviceWrapper(root).open_url("https://example.com")
            )
            await asyncio.sleep(0)
            assert not pending.done()
            assert await independent.input_text("independent")
        await pending
        assert root.opened == "https://example.com"

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_cancelling_active_input_releases_the_root_lock():
    async def run():
        started = asyncio.Event()

        class BlockingDevice(MemoryDevice):
            async def tap(self, x: int, y: int) -> None:
                started.set()
                await asyncio.Event().wait()

        root = BlockingDevice()
        tapping = asyncio.create_task(DeviceWrapper(root).tap(1, 1))
        await started.wait()
        tapping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tapping
        assert await root.input_text("after cancellation")

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_sensitive_parameter_default_is_rejected_at_declaration():
    with pytest.raises(TypeError, match="Sensitive.*default"):

        class Unsafe(Extension):
            @action(sensitive_parameters=frozenset({"password"}))
            async def login(self, password: str = "TOP-SECRET") -> None:
                """Use credentials."""


class SecretInput(BaseModel):
    token: str = Field(default="NESTED-SECRET", examples=["EXAMPLE-SECRET"])
    kind: Literal["CONST-SECRET"] = "CONST-SECRET"


def test_sensitive_schemas_strip_examples_defaults_and_const_in_definitions():
    class Safe(Extension):
        @action(sensitive_parameters=frozenset({"credentials"}))
        async def login(
            self,
            credentials: Annotated[
                SecretInput, Field(examples=[{"token": "ROOT-SECRET"}])
            ],
            count: int = 3,
        ) -> None:
            """Use credentials without publishing examples."""

    for audience in ("python", "operator", "cli"):
        schema = Safe().actions.list(audience=audience)[0].input_schema
        assert "SECRET" not in json.dumps(schema)
        assert schema["properties"]["count"]["default"] == 3


@pytest.mark.parametrize("bound", [False, True])
def test_sensitive_action_redacts_sdk_logs_and_nested_sensitive_calls(caplog, bound):
    logger = logging.getLogger("review.mock_sdk")

    class Resource(Extension):
        @action(sensitive_parameters=frozenset({"password"}))
        async def login(self, password: str) -> None:
            """Make an SDK request that logs its password."""
            logger.warning("request password=%s", password)
            if hasattr(self, "outer_value"):
                logger.warning("nested outer=%s inner=%s", self.outer_value, password)

        @action(sensitive_parameters=frozenset({"credentials"}))
        async def outer(self, credentials: dict[str, str]) -> None:
            """Keep outer secrets active through an inner sensitive action."""
            self.outer_value = credentials["password"]
            await self.login("INNER-CANARY")
            del self.outer_value
            logger.warning("outer password=%s", credentials["password"])

    async def scenario():
        resource = Resource()
        phone = MemoryDevice()
        if bound:
            await resource.bind(phone)
        await resource.login("DIRECT-CANARY")
        await resource.outer({"password": "OUTER-CANARY"})
        await resource.actions.call("extension.login", {"password": "CATALOG-CANARY"})
        await phone.close()
        await resource.close()

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())
    assert len(caplog.records) == 5
    assert all("CANARY" not in record.getMessage() for record in caplog.records)
    assert all("***" in record.getMessage() for record in caplog.records)
    # The context must not permanently redact unrelated later log records.
    logger.warning("outside password=%s", "DIRECT-CANARY")
    assert "DIRECT-CANARY" in caplog.records[-1].getMessage()


class SecretChoice(str, Enum):
    FIRST = "ENUM-CANARY-ONE"
    SECOND = "ENUM-CANARY-TWO"


class SecretConstraints(BaseModel):
    literal: Literal["LITERAL-CANARY-ONE", "LITERAL-CANARY-TWO"]
    choice: SecretChoice
    token: Annotated[str, Field(pattern="PATTERN-CANARY")]


def test_sensitive_enum_registers_the_underlying_value(caplog):
    class Resource(Extension):
        @action(sensitive_parameters=frozenset({"choice"}))
        async def use(self, choice: SecretChoice) -> None:
            """Simulate an SDK serializing an enum into its request log."""
            logging.getLogger("review.mock_sdk").warning(
                "request value=%s", choice.value
            )

    asyncio.run(Resource().use(SecretChoice.FIRST))
    assert "ENUM-CANARY" not in caplog.text
    assert "***" in caplog.text


def test_sensitive_schema_strips_enum_literal_and_pattern_from_referenced_defs():
    class Resource(Extension):
        @action(sensitive_parameters=frozenset({"secret"}))
        async def use(self, secret: SecretConstraints) -> None:
            """Accept constrained secret inputs without publishing their values."""

    import json

    for audience in ("python", "operator", "cli"):
        schema = Resource().actions.list(audience=audience)[0].input_schema
        assert "CANARY" not in json.dumps(dict(schema))
        assert "enum" not in json.dumps(dict(schema))
        assert "pattern" not in json.dumps(dict(schema))


def test_wrapper_copy_and_pickle_are_explicitly_rejected():
    wrapper = DeviceWrapper(MemoryDevice())
    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="copied or pickled"):
            operation(wrapper)
    partial = DeviceWrapper.__new__(DeviceWrapper)
    with pytest.raises(AttributeError):
        partial.__getattr__("missing")


def test_device_has_no_typed_get_api():
    with pytest.raises(AttributeError):
        _ = MemoryDevice().get


@pytest.mark.parametrize(
    ("role", "default", "attribute"),
    [("phone_number", "sms", "number"), ("mailbox", "email", "mailbox")],
)
def test_resource_lookup_uses_declared_roles_and_default_names(
    role, default, attribute
):
    selected_role = role

    class Resource(Extension):
        role = selected_role

    async def scenario():
        phone = MemoryDevice()
        first = Resource(name="first")
        second = Resource(name="second")
        preferred = Resource(name=default)
        unrelated = Extension(name=default)
        await unrelated.bind(phone)
        with pytest.raises(DeviceOperationNotSupported, match="No"):
            getattr(phone, attribute)
        await first.bind(phone)
        assert getattr(phone, attribute) is first
        await second.bind(phone)
        with pytest.raises(DeviceOperationNotSupported, match="Multiple"):
            getattr(phone, attribute)
        await unrelated.unbind()
        await preferred.bind(phone)
        assert getattr(phone, attribute) is preferred
        await phone.close()

    asyncio.run(scenario())


def test_cloud_task_reference_validation_checks_fields_not_concrete_class():
    @dataclass(frozen=True)
    class Reference:
        package: str
        name: str

    reference = Reference("app", "work")
    inputs = CloudTaskInputs((cast(CloudCredentialReference, reference),))
    assert inputs.credential_references[0] is reference
    with pytest.raises(TypeError):
        CloudTaskInputs((cast(CloudCredentialReference, object()),))
