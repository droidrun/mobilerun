"""Independent-review regressions for extension safety and lifecycle."""

import asyncio
import copy
import json
import pickle
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field
from test_proxy_extensions import ProxyMemory

from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.extensions import Extension, action
from mobilerun.core.extensions.registry import Binding
from mobilerun.core.proxy import LocalProxy, ProxyStatus
from mobilerun.errors import ExtensionActionError, ExtensionNotBound
from tests.support.devices import MemoryDevice
from tests.support.extensions import check_extension_contract


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


def test_wrapper_copy_and_pickle_are_explicitly_rejected():
    wrapper = DeviceWrapper(MemoryDevice())
    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="copied or pickled"):
            operation(wrapper)
    partial = DeviceWrapper.__new__(DeviceWrapper)
    with pytest.raises(AttributeError):
        partial.__getattr__("missing")


@pytest.mark.parametrize("force", [False, True])
def test_failed_detach_with_unreadable_status_is_uncertain(force):
    from unittest.mock import AsyncMock

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


def test_device_has_no_typed_get_api():
    with pytest.raises(AttributeError):
        _ = MemoryDevice().get
