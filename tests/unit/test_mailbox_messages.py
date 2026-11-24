"""Mailbox history uses the cloud message list without purchasing resources."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from click.testing import CliRunner
from mobilerun_sdk.types.mailboxes.message_list_response import (
    Item,
    MessageListResponse,
)

from mobilerun.cli.app import cli
from mobilerun.clients.cloud import mailboxes
from mobilerun.models.billing import Billing


def message(id="msg-1"):
    return Item(
        id=id,
        mailboxId="box-1",
        fromAddress="a@example.com",
        hasOtp=True,
        receivedAt="2025-01-01T00:00:00Z",
        subject="Your code",
        preview="Hello",
    )


def test_mailbox_messages_paginates_and_uses_existing_client():
    listing = AsyncMock(
        side_effect=[
            MessageListResponse(items=[message()], nextCursor="next"),
            MessageListResponse(items=[message("msg-2")]),
        ]
    )
    client = SimpleNamespace(
        mailboxes=SimpleNamespace(messages=SimpleNamespace(list=listing))
    )
    result = asyncio.run(
        mailboxes.messages("box-1", billing=Billing.NO_COMMERCE, client=client)
    )
    assert [item.id for item in result] == ["msg-1", "msg-2"]
    assert listing.await_args_list[1].kwargs["cursor"] == "next"


def test_mailbox_messages_cli_json_and_text(monkeypatch):
    listing = AsyncMock(return_value=[message()])
    monkeypatch.setattr(mailboxes, "messages", listing, raising=False)
    runner = CliRunner()
    result = runner.invoke(cli, ["mailbox", "messages", "box-1", "--json"])
    assert result.exit_code == 0, result.output
    entry = json.loads(result.stdout)[0]
    assert entry["from_address"] == "a@example.com"
    assert entry["has_otp"] is True
    assert entry["preview"] == "Hello"
    result = runner.invoke(cli, ["mailbox", "messages", "box-1"])
    assert result.exit_code == 0, result.output
    assert "a@example.com" in result.output
    assert "Your code" in result.output
    assert "Hello" in result.output


def test_mailbox_messages_cli_empty(monkeypatch):
    monkeypatch.setattr(mailboxes, "messages", AsyncMock(return_value=[]))
    runner = CliRunner()
    result = runner.invoke(cli, ["mailbox", "messages", "box-1", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == []
    result = runner.invoke(cli, ["mailbox", "messages", "box-1"])
    assert result.exit_code == 0
    assert "messages" in result.output.lower()
