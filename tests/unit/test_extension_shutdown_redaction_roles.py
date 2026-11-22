"""Review regressions for shutdown, sensitive SDK logging, schemas and roles."""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal, cast

import pytest
from pydantic import BaseModel, Field

from mobilerun.core.extensions import Extension, action
from mobilerun.errors import DeviceOperationNotSupported, ExtensionClosed
from mobilerun.models import CloudCredentialReference, CloudTaskInputs
from tests.support.devices import MemoryDevice


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
