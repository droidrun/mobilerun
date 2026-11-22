"""Useful action failures without backend messages or sensitive cause chains."""

import asyncio
import logging
import traceback

import pytest

from mobilerun.core.extensions import Extension, action
from mobilerun.errors import (
    CloudServiceUnavailable,
    CredentialFieldNotFound,
    ExtensionActionError,
)


@pytest.mark.parametrize("timeout", [180, 180.0])
def test_timeout_reports_type_and_configured_nonsensitive_timeout(timeout, caplog):
    caplog.set_level(logging.INFO, logger="mobilerun")

    class Mailbox(Extension):
        @action
        async def wait_for_code(self, timeout: float = 180) -> str:
            """Wait for a test verification code."""
            raise TimeoutError("BACKEND-CANARY")

    async def run():
        mailbox = Mailbox(name="email")
        for invoke in (
            mailbox.wait_for_code,
            lambda: mailbox.wait_for_code(timeout),
            lambda: mailbox.actions.call("email.wait_for_code", {"timeout": timeout}),
            lambda: mailbox.actions.call("email.wait_for_code", {}),
        ):
            with pytest.raises(ExtensionActionError) as raised:
                await invoke()
            assert (
                str(raised.value)
                == "Action email.wait_for_code failed: TimeoutError (timed out after 180s)"
            )
            assert raised.value.cause_type == "TimeoutError"
            assert raised.value.__cause__ is None
            assert "BACKEND-CANARY" not in "".join(
                traceback.format_exception(raised.value)
            )

    asyncio.run(run())


def test_timeout_without_a_configured_timeout_does_not_guess(caplog):
    caplog.set_level(logging.INFO, logger="mobilerun")

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Fail with an unknown backend timeout."""
            raise TimeoutError("PRIVATE-CANARY")

    with pytest.raises(ExtensionActionError) as raised:
        asyncio.run(Resource().run())
    assert str(raised.value) == "Action extension.run failed: TimeoutError (timed out)"
    assert raised.value.cause_type == "TimeoutError"


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_sensitive_action_never_exposes_backend_message_or_chain(level, caplog):
    caplog.set_level(level, logger="mobilerun")
    canary = "PRIVATE-CANARY"

    class Vault(Extension):
        @action(sensitive_parameters=frozenset({"password"}))
        async def login(self, password: str, timeout: float = 180) -> None:
            """Perform a sensitive SDK operation."""
            raise TimeoutError(f"secret={password}")

    with pytest.raises(ExtensionActionError) as raised:
        asyncio.run(Vault().login(canary))
    assert raised.value.cause_type == "TimeoutError"
    assert "180s" in str(raised.value)
    assert raised.value.__cause__ is None
    assert "PRIVATE-CANARY" not in "".join(traceback.format_exception(raised.value))


def test_sensitive_timeout_is_not_added_to_message(caplog):
    caplog.set_level(logging.INFO, logger="mobilerun")

    class Resource(Extension):
        @action(sensitive_parameters=frozenset({"timeout"}))
        async def run(self, timeout: float) -> None:
            """Treat the timeout itself as confidential."""
            raise TimeoutError("PRIVATE-CANARY")

    with pytest.raises(ExtensionActionError) as raised:
        asyncio.run(Resource().run(987654))
    assert "TimeoutError (timed out)" in str(raised.value)
    assert "987654" not in str(raised.value)


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_cloud_error_propagates_unchanged(level, caplog):
    caplog.set_level(level, logger="mobilerun")
    original = CloudServiceUnavailable(operation="List SMS messages", status=503)

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Exercise opt-in debug diagnostics."""
            raise original

    with pytest.raises(CloudServiceUnavailable) as raised:
        asyncio.run(Resource().run())
    assert raised.value is original


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_nonsensitive_debug_action_chains_only_when_debug_enabled(level, caplog):
    caplog.set_level(level, logger="mobilerun")
    original = RuntimeError("BACKEND-CANARY")

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Exercise opt-in debug diagnostics for ordinary exceptions."""
            raise original

    with pytest.raises(ExtensionActionError) as raised:
        asyncio.run(Resource().run())
    assert raised.value.cause_type == "RuntimeError"
    assert "RuntimeError" in str(raised.value)
    assert "BACKEND-CANARY" not in str(raised.value)
    assert raised.value.__cause__ is (original if level == logging.DEBUG else None)


def test_nested_nonsensitive_action_cannot_escape_sensitive_debug_scope(caplog):
    caplog.set_level(logging.DEBUG, logger="mobilerun")
    canary = "PRIVATE-CANARY"

    class Resource(Extension):
        @action
        async def inner(self) -> None:
            """Fail inside the parent sensitive scope."""
            raise RuntimeError(canary)

        @action(sensitive_parameters=frozenset({"password"}))
        async def outer(self, password: str) -> None:
            """Delegate within a sensitive call."""
            await self.inner()

    with pytest.raises(ExtensionActionError) as raised:
        asyncio.run(Resource().outer(canary))
    assert raised.value.cause_type == "RuntimeError"
    assert raised.value.__cause__ is None
    assert "PRIVATE-CANARY" not in "".join(traceback.format_exception(raised.value))


def test_value_safe_error_still_propagates_unchanged(caplog):
    caplog.set_level(logging.DEBUG, logger="mobilerun")
    original = CredentialFieldNotFound("Credential field not found")

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Expose a public value-safe typed failure."""
            raise original

    with pytest.raises(CredentialFieldNotFound) as raised:
        asyncio.run(Resource().run())
    assert raised.value is original
