"""The shared retry policy uses guard failures, not failed transport calls."""

import asyncio
import os
import select
import signal
import subprocess
import sys
from time import monotonic
from types import SimpleNamespace

import pytest
from prompt_toolkit import PromptSession

from mobilerun.clients.cloud import billing as policy
from mobilerun.errors import BillingError, BillingRequired
from mobilerun.errors.base import is_value_safe
from mobilerun.models.billing import Billing


@pytest.mark.parametrize(
    "mode,tty,yes,answer,expected",
    [
        (Billing.PAID, True, False, None, [Billing.PAID]),
        (Billing.INCLUDED, True, False, "y", [Billing.INCLUDED, Billing.PAID]),
        (Billing.INCLUDED, True, False, "", [Billing.INCLUDED]),
        (Billing.INCLUDED, False, False, None, [Billing.INCLUDED]),
        (Billing.NO_COMMERCE, False, True, None, [Billing.NO_COMMERCE, Billing.PAID]),
    ],
)
def test_retry_only_runs_after_consent(
    monkeypatch, capsys, mode, tty, yes, answer, expected
):
    attempts = []
    effects = []
    replies = []

    @policy.requires(Billing.PAID, operation="Buy Connect proxy")
    async def purchase(*, billing):
        effects.append(billing)
        return "proxy-id"

    def operation(billing):
        attempts.append(billing)
        return purchase(billing=billing)

    async def respond(self, message):
        assert answer is not None, "unexpected billing prompt"
        replies.append(answer)
        sys.stderr.write(message)
        return answer

    monkeypatch.setattr(sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: tty)
    monkeypatch.setattr(PromptSession, "prompt_async", respond)
    if expected[-1] is Billing.PAID:
        assert asyncio.run(policy.with_billing_retry(operation, mode, yes=yes)) == (
            "proxy-id",
            Billing.PAID,
        )
        assert effects == [Billing.PAID]
    else:
        with pytest.raises(BillingRequired) as caught:
            asyncio.run(policy.with_billing_retry(operation, mode, yes=yes))
        error = caught.value
        assert (error.needed, error.given, error.operation) == (
            Billing.PAID,
            mode,
            "Buy Connect proxy",
        )
        assert is_value_safe(error)
        assert all(
            hint in str(error)
            for hint in ("--yes", "yes=True", "MOBILERUN_BILLING=paid")
        )
        assert not effects
    assert attempts == expected
    assert replies == ([] if answer is None else [answer])
    assert ("Buy Connect proxy" in capsys.readouterr().err) is (answer is not None)


@pytest.mark.parametrize("second_failure", [BillingRequired, BillingError])
def test_retry_never_catches_second_failure(second_failure):
    attempts = []
    failure = (
        BillingRequired(
            needed=Billing.PAID, given=Billing.INCLUDED, operation="Buy Connect proxy"
        )
        if second_failure is BillingRequired
        else BillingError(code="billing_override")
    )

    async def operation(billing):
        attempts.append(billing)
        raise failure

    with pytest.raises(second_failure) as caught:
        asyncio.run(policy.with_billing_retry(operation, Billing.INCLUDED, yes=True))
    assert caught.value is failure
    assert attempts == (
        [Billing.INCLUDED, Billing.PAID]
        if second_failure is BillingRequired
        else [Billing.INCLUDED]
    )


def test_guard_error_never_renders_unknown_operation_text():
    error = BillingRequired(
        needed=Billing.PAID, given=Billing.INCLUDED, operation="private value"
    )
    assert "private value" not in str(error)
    assert "private value" not in repr(error)
    assert is_value_safe(error)


def test_nested_guard_rejects_second_attempt_with_explicit_billing_hint():
    attempts = []
    mutations = []

    @policy.requires(Billing.PAID, operation="Send SMS")
    async def send(*, billing):
        mutations.append(billing)

    @policy.requires(Billing.INCLUDED, operation="Run cloud task")
    async def prepare(*, billing):
        await send(billing=billing)

    def operation(mode):
        attempts.append(mode)
        return prepare(billing=mode)

    with pytest.raises(BillingRequired) as caught:
        asyncio.run(policy.with_billing_retry(operation, Billing.NO_COMMERCE, yes=True))
    error = caught.value
    assert (error.given, error.needed, error.operation) == (
        Billing.INCLUDED,
        Billing.PAID,
        "Send SMS",
    )
    assert attempts == [Billing.NO_COMMERCE, Billing.INCLUDED]
    assert not mutations
    assert 'Pass --billing paid (CLI) or billing="paid" (Python).' in str(error)
    assert str(error).index("--billing paid") < str(error).index("--yes")


def test_sdk_prompt_sigint_exits_without_enter():
    pty = pytest.importorskip("pty")
    master, slave = pty.openpty()
    script = """
import asyncio
from mobilerun.clients.cloud.billing import requires, with_billing_retry
from mobilerun.models.billing import Billing

@requires(Billing.PAID)
async def purchase(*, billing):
    raise AssertionError('purchase must not run')

try:
    asyncio.run(with_billing_retry(lambda mode: purchase(billing=mode), Billing.INCLUDED))
except KeyboardInterrupt:
    raise SystemExit(130)
"""
    try:
        with subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=slave,
            stdout=subprocess.PIPE,
            stderr=slave,
        ) as process:
            os.close(slave)
            slave = None
            try:
                prompt = b""
                deadline = monotonic() + 5
                while b"[y/N]" not in prompt:
                    assert select.select(
                        [master], [], [], max(0, deadline - monotonic())
                    )[0], prompt
                    chunk = os.read(master, 4096)
                    assert chunk, prompt
                    prompt += chunk
                process.send_signal(signal.SIGINT)
                process.communicate(timeout=3)
                assert process.returncode == 130
            finally:
                if process.poll() is None:
                    process.kill()
    finally:
        os.close(master)
        if slave is not None:
            os.close(slave)


@pytest.mark.parametrize(
    "given,needed", [("no_commerce", Billing.INCLUDED), ("included", Billing.PAID)]
)
def test_retry_uses_exact_guard_minimum(monkeypatch, capsys, given, needed):
    attempts = []

    @policy.requires(needed, operation="Create cloud phone")
    async def create(*, billing):
        return billing

    def operation(billing):
        attempts.append(billing)
        return create(billing=billing)

    async def respond(self, message):
        sys.stderr.write(message)
        return "yes"

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr(PromptSession, "prompt_async", respond)
    assert asyncio.run(policy.with_billing_retry(operation, given)) == (needed, needed)
    assert attempts[0] is Billing(given) and attempts[1] is needed
    prompt = capsys.readouterr().err
    assert f"{needed.value} billing" in prompt
    assert ("charges may apply" in prompt) is (needed is Billing.PAID)


def test_truthy_yes_does_not_authorize_noninteractive_retry():
    attempts = []

    @policy.requires(Billing.PAID, operation="Send SMS")
    async def send(*, billing):
        pytest.fail("unapproved charge")

    def operation(billing):
        attempts.append(billing)
        return send(billing=billing)

    with pytest.raises(BillingRequired):
        asyncio.run(
            policy.with_billing_retry(
                operation, Billing.INCLUDED, yes="false", is_interactive=lambda: False
            )
        )
    assert attempts == [Billing.INCLUDED]


@pytest.mark.parametrize(
    "stdin_tty,stderr_tty", [(None, True), (True, None), (True, False)]
)
def test_sdk_does_not_prompt_without_visible_terminal(
    monkeypatch, stdin_tty, stderr_tty
):
    @policy.requires(Billing.PAID, operation="Send SMS")
    async def send(*, billing):
        pytest.fail("unapproved charge")

    def unexpected_prompt(*args, **kwargs):
        pytest.fail("billing prompt requires interactive stdin and stderr")

    monkeypatch.setattr("prompt_toolkit.PromptSession", unexpected_prompt)
    with monkeypatch.context() as patch:
        for name, tty in (("stdin", stdin_tty), ("stderr", stderr_tty)):
            patch.setattr(
                sys,
                name,
                None if tty is None else SimpleNamespace(isatty=lambda tty=tty: tty),
            )
        with pytest.raises(BillingRequired, match="--billing paid"):
            asyncio.run(
                policy.with_billing_retry(
                    lambda mode: send(billing=mode), Billing.INCLUDED
                )
            )
