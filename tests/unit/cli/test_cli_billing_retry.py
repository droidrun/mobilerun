"""Exercise billing consent through one fake Click command."""

import asyncio
import importlib
import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli as app
from mobilerun.cli.base import menu, safety
from mobilerun.cli.base.billing import billing_option, invoke_with_billing
from mobilerun.cli.devices.base import selection
from mobilerun.clients.cloud import mailboxes, numbers, proxies
from mobilerun.clients.cloud.billing import requires
from mobilerun.core.devices.base import Device
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.extensions import action
from mobilerun.core.phone_number import CloudPhoneNumber
from mobilerun.models.billing import Billing, Plan


class TTYRunner(CliRunner):
    def __init__(self, stdin, stdout, stderr):
        super().__init__()
        self.ttys = (stdin, stdout, stderr)

    @contextmanager
    def isolation(self, *args, **kwargs):
        with super().isolation(*args, **kwargs) as streams:
            for stream, tty in zip(
                (sys.stdin, sys.stdout, sys.stderr), self.ttys, strict=True
            ):
                stream.isatty = lambda tty=tty: tty
            yield streams


@pytest.mark.parametrize(
    "mode,ttys,yes,answer,exit_code,attempts",
    [
        (Billing.PAID, (True, True, True), False, None, 0, [Billing.PAID]),
        (
            Billing.INCLUDED,
            (True, False, True),
            False,
            True,
            0,
            [Billing.INCLUDED, Billing.PAID],
        ),
        (Billing.INCLUDED, (True, True, True), False, False, 1, [Billing.INCLUDED]),
        (Billing.INCLUDED, (True, True, False), False, None, 1, [Billing.INCLUDED]),
        (Billing.INCLUDED, (False, True, True), False, None, 1, [Billing.INCLUDED]),
        (
            Billing.NO_COMMERCE,
            (False, False, False),
            True,
            None,
            0,
            [Billing.NO_COMMERCE, Billing.PAID],
        ),
    ],
)
def test_cli_billing_retry(monkeypatch, mode, ttys, yes, answer, exit_code, attempts):
    attempted = []
    purchased = []
    prompts = []

    @requires(Billing.PAID, operation="Buy Connect proxy")
    async def purchase(*, billing):
        purchased.append(billing)
        return "proxy-id"

    def operation(billing):
        attempted.append(billing)
        return purchase(billing=billing)

    async def prompt(message, *, default):
        assert answer is not None, "unexpected billing prompt"
        assert "Buy Connect proxy" in message
        assert default is False
        prompts.append(message)
        click.echo(message)
        return answer

    monkeypatch.setattr(safety, "confirm_prompt_async", prompt)

    @click.group(cls=menu.MenuGroup)
    def cli():
        pass

    @cli.command()
    @click.option("--yes", is_flag=True)
    @billing_option
    def buy(billing, yes):
        result, effective = asyncio.run(
            invoke_with_billing(operation, Billing(billing), yes=yes)
        )
        click.echo(f"{result} {effective}")

    result = TTYRunner(*ttys).invoke(
        cli, ["buy", "--billing", mode.value, *(["--yes"] if yes else [])]
    )
    assert result.exit_code == exit_code, result.output
    assert attempted == attempts
    assert len(prompts) == (answer is not None)
    if attempts[-1] is Billing.PAID:
        assert result.stdout == "proxy-id paid\n"
        assert purchased == [Billing.PAID]
    else:
        assert not result.stdout
        assert not purchased
    if answer is False:
        assert "Cancelled" in result.stderr
    elif exit_code:
        assert all(
            hint in result.stderr
            for hint in ("--yes", "yes=True", "MOBILERUN_BILLING=paid")
        )


@pytest.mark.parametrize(
    "command,mode,interactive,yes,answer,runs,exit_code",
    [
        (command, *case)
        for command in (
            "number create",
            "mailbox create",
            "proxy buy",
            "number send",
            "resume",
        )
        for case in (
            ("included", True, False, True, 1, 0),
            ("included", True, False, False, 0, 1),
            ("included", False, True, None, 1, 0),
            ("included", False, False, None, 0, 1),
            ("no_commerce", False, True, None, 1, 0),
        )
    ]
    + [
        (command, "paid", True, False, None, 1, 0)
        for command in ("number create", "mailbox create", "proxy buy")
    ]
    + [
        (command, "paid", interactive, yes, answer, runs, exit_code)
        for command in ("number send", "resume")
        for interactive, yes, answer, runs, exit_code in (
            (True, False, True, 1, 0),
            (True, False, False, 0, 1),
            (True, True, None, 1, 0),
            (False, True, None, 1, 0),
            (False, False, None, 0, 1),
        )
    ],
)
def test_resource_cli_retries_only_guarded_operation(
    monkeypatch, command, mode, interactive, yes, answer, runs, exit_code
):
    data = NS(
        number_id="num",
        state="active",
        checkout_url=None,
        id="mail",
        status="active",
        address="mail@example.test",
        phone_number="+15550100",
    )
    mutate = AsyncMock(return_value=NS(data=data))
    details = AsyncMock(return_value=NS(data=data))
    metadata = AsyncMock(return_value=NS(billing_strategy=Plan.MINUTE))
    prompts = []
    client_entries = []

    @asynccontextmanager
    async def client(*args, **kwargs):
        client_entries.append(kwargs["billing"])
        yield NS(
            devices=NS(retrieve=metadata, resume=mutate),
            numbers=NS(create=mutate, retrieve=details, messages=NS(send=mutate)),
            mailboxes=NS(create=mutate, retrieve=details),
            connect=NS(proxies=NS(buy=mutate)),
        )

    async def confirm(message, *, default):
        mutate.assert_not_awaited()
        assert len(client_entries) == (1 if command == "resume" else 0)
        assert answer is not None, "unexpected billing prompt"
        assert "charges may apply" in message and default is False
        prompts.append(message)
        return answer

    monkeypatch.setattr(safety, "confirm_prompt_async", confirm)
    monkeypatch.setenv("MOBILERUN_BILLING", mode)
    for module in (
        numbers,
        mailboxes,
        proxies,
        importlib.import_module("mobilerun.cli.devices.cloud.resume"),
    ):
        monkeypatch.setattr(module, "cloud_client", client)
    args = command.split()
    if command == "number create":
        args += ["--country", "US", "--plan", "rent"]
    elif command == "mailbox create":
        args += ["--plan", "rent"]
    elif command == "proxy buy":
        mutate.return_value = NS(id="proxy", status="active", payment_url=None)
        args += ["--country", "US"]
    elif command == "number send":
        args += ["number-id", "+15550100", "hello"]
    else:
        args = ["device", "cloud", "resume", "aa111111-1111-1111-1111-111111111111"]

    result = TTYRunner(interactive, interactive, interactive).invoke(
        app, [*args, *(["--yes"] if yes else [])]
    )
    assert result.exit_code == exit_code, result.output
    assert len(prompts) == (answer is not None)
    assert mutate.await_count == runs
    assert metadata.await_count == (command == "resume")
    if command != "resume":
        assert client_entries == [Billing.PAID] * (runs + details.await_count)
    if answer is False:
        assert "Cancelled" in result.stderr
    elif exit_code and mode == "paid":
        assert "without confirmation; pass --yes" in result.stderr
    elif exit_code:
        assert all(
            hint in result.stderr
            for hint in ("--yes", "yes=True", "MOBILERUN_BILLING=paid")
        )


@pytest.mark.parametrize("mode", ["paid", "included", "no_commerce"])
@pytest.mark.parametrize("interactive", [True, False])
def test_subscription_resume_never_prompts(
    monkeypatch: pytest.MonkeyPatch, mode: str, interactive: bool
) -> None:
    metadata = AsyncMock(return_value=NS(billing_strategy=Plan.SUBSCRIPTION))
    resume = AsyncMock()
    prompt = AsyncMock(side_effect=AssertionError("unexpected billing prompt"))

    @asynccontextmanager
    async def client(*, base_url: str | None, billing: Billing) -> AsyncIterator[NS]:
        yield NS(devices=NS(retrieve=metadata, resume=resume))

    module = importlib.import_module("mobilerun.cli.devices.cloud.resume")
    monkeypatch.setattr(module, "cloud_client", client)
    monkeypatch.setattr(safety, "confirm_prompt_async", prompt)
    device_id = "aa111111-1111-1111-1111-111111111111"
    result = TTYRunner(interactive, False, interactive).invoke(
        app, ["device", "cloud", "resume", device_id, "--billing", mode]
    )

    assert result.exit_code == 0, result.output
    prompt.assert_not_awaited()
    metadata.assert_awaited_once_with(device_id)
    resume.assert_awaited_once_with(device_id)


@pytest.mark.parametrize("alias", ["sms", "work"])
@pytest.mark.parametrize(
    "mode,yes,reply,exit_code,sends",
    [
        ("paid", False, True, 0, 1),
        ("paid", False, False, 1, 0),
        ("paid", True, None, 0, 1),
        ("paid", False, None, 1, 0),
        ("included", True, None, 0, 1),
        ("included", False, True, 0, 1),
        ("included", False, False, 1, 0),
        ("included", False, None, 1, 0),
    ],
)
def test_bound_cloud_sms_retries_only_sdk_send(
    monkeypatch, alias, mode, yes, reply, exit_code, sends
):
    class BoundNumber(CloudPhoneNumber):
        async def check_bind(self, device):
            return ()

        async def on_bind(self, device):
            pass

    attempts = []
    send = numbers.send

    async def track_send(*args, billing, **kwargs):
        attempts.append(billing)
        return await send(*args, billing=billing, **kwargs)

    monkeypatch.setattr(numbers, "send", track_send)
    transport = AsyncMock(
        return_value=numbers.MessageSendResponse.model_validate(
            {
                "data": {
                    "id": "sms-id",
                    "createdAt": "2026-01-01T00:00:00Z",
                    "occurredAt": "2026-01-01T00:00:00Z",
                    "direction": "outbound",
                    "status": "queued",
                }
            }
        )
    )
    sdk = NS(numbers=NS(messages=NS(send=transport)))
    number = BoundNumber("number-id", billing=mode, client=sdk, name=alias)
    phone = RecordingDevice(Device())
    phone.close = AsyncMock()
    asyncio.run(number.bind(phone))
    open_phone = AsyncMock(return_value=phone)
    monkeypatch.setattr(selection, "connect_target", open_phone)
    # The extension's billing snapshot, not the CLI environment, controls SMS.
    monkeypatch.setenv("MOBILERUN_BILLING", "no_commerce")

    async def confirm(message, *, default):
        assert reply is not None
        assert "Send SMS" in message and not default
        transport.assert_not_called()
        return reply

    prompt = AsyncMock(side_effect=confirm)
    monkeypatch.setattr(safety, "confirm_prompt_async", prompt)
    args = [
        "device",
        "action",
        "call",
        f"{alias}.send",
        "aa111111-1111-1111-1111-111111111111",
        "--arg",
        "to=5550100",
        "--arg",
        "text=hello",
        "--json",
    ]
    interactive = reply is not None or (yes and mode == "paid")
    result = TTYRunner(interactive, False, interactive).invoke(
        app, [*args, *(["--yes"] if yes else [])]
    )
    assert result.exit_code == exit_code, result.output
    assert prompt.await_count == (reply is not None)
    expected_attempts = [Billing(mode)] if mode != "paid" or sends else []
    if sends and mode != "paid":
        expected_attempts.append(Billing.PAID)
    assert attempts == expected_attempts
    open_phone.assert_awaited_once()
    phone.close.assert_awaited_once()
    if sends:
        assert json.loads(result.stdout)["data"]["id"] == "sms-id"
        transport.assert_awaited_once_with("number-id", to="5550100", body="hello")
        assert list(phone.recorded_actions) == [
            {
                "action_type": "extension_call",
                "name": f"{alias}.send",
                "arguments": {"to": "5550100", "text": "hello"},
            }
        ]
    else:
        transport.assert_not_called()
        assert not phone.recorded_actions
        assert ("Cancelled" if reply is False else "--yes") in result.stderr


@pytest.mark.parametrize(
    "override", ["plain", "decorated", "mixed_inheritance", "wraps", "same_signature"]
)
def test_cloud_number_with_replaced_send_still_requires_paid_confirmation(
    monkeypatch, override
):
    sent = []

    async def send(self, to: str, text: str) -> str:
        """Send via a different provider."""
        sent.append((to, text))
        return "sent"

    @wraps(CloudPhoneNumber.send)
    async def wrapped_send(self, to, text, **kwargs):
        assert not kwargs
        sent.append((to, text))
        return numbers.MessageSendResponse.model_validate(
            {
                "data": {
                    "id": "replacement-sms",
                    "createdAt": "2026-01-01T00:00:00Z",
                    "occurredAt": "2026-01-01T00:00:00Z",
                    "direction": "outbound",
                    "status": "queued",
                }
            }
        )

    async def typed_send(
        self: "CloudPhoneNumber", to: str, text: str
    ) -> numbers.MessageSendResponse:
        """Replace the provider while preserving the typed signature."""
        return await wrapped_send(self, to, text)

    typed_send.__name__ = "send"
    replacement = {
        "wraps": wrapped_send,
        "decorated": action(send),
        "same_signature": typed_send,
    }.get(override, send)

    class Provider(CloudPhoneNumber):
        send = replacement

        async def check_bind(self, device):
            return ()

        async def on_bind(self, device):
            pass

    class InheritedSend(CloudPhoneNumber):
        pass

    class MultipleBases(InheritedSend, Provider):
        pass

    phone = Device()
    phone.close = AsyncMock()
    number_type = MultipleBases if override == "mixed_inheritance" else Provider
    asyncio.run(number_type("number-id", name="work").bind(phone))
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=phone))
    args = [
        "device",
        "action",
        "call",
        "work.send",
        "aa111111-1111-1111-1111-111111111111",
        "--arg",
        "to=5550100",
        "--arg",
        "text=hello",
    ]
    monkeypatch.setenv("MOBILERUN_BILLING", "included")
    result = CliRunner().invoke(app, [*args, "--yes"])
    assert sent == []
    assert result.exit_code == 1 and "MOBILERUN_BILLING=paid" in result.stderr
    monkeypatch.setenv("MOBILERUN_BILLING", "paid")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1 and "--yes" in result.stderr
    assert sent == []
    prompt = AsyncMock(return_value=False)
    monkeypatch.setattr(safety, "confirm_prompt_async", prompt)
    result = TTYRunner(True, False, True).invoke(app, args)
    assert result.exit_code == 1 and "Cancelled" in result.stderr
    prompt.assert_awaited_once()
    assert sent == []
    result = CliRunner().invoke(app, [*args, "--yes"])
    assert result.exit_code == 0, result.stderr
    if override in {"wraps", "same_signature"}:
        assert json.loads(result.stdout)["data"]["id"] == "replacement-sms"
    else:
        assert result.stdout.strip() == "sent"
    assert sent == [("5550100", "hello")]
