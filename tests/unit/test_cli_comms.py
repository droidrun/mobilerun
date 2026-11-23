"""CLI communications commands use safe billing unless explicitly opted in."""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

import mobilerun.cli.mailbox.group as mailbox_commands
import mobilerun.cli.phone_number.group as number_commands
from mobilerun.cli.app import cli
from mobilerun.cli.mailbox.group import Mailbox
from mobilerun.cli.phone_number.group import PhoneNumber


@pytest.fixture(autouse=True)
def fake_resource_retrieval(monkeypatch):
    data = NS(
        id="resource-id",
        phone_number=None,
        address=None,
        state="active",
        status="active",
    )
    monkeypatch.setattr(
        number_commands.cloud_numbers, "retrieve", AsyncMock(return_value=NS(data=data))
    )
    monkeypatch.setattr(
        mailbox_commands.cloud_mailboxes,
        "retrieve",
        AsyncMock(return_value=NS(data=data)),
    )


@pytest.mark.parametrize(
    "group,resource", [("number", PhoneNumber), ("mailbox", Mailbox)]
)
@pytest.mark.parametrize("allowed", [False, True])
def test_cli_create_billing(monkeypatch, group, resource, allowed):
    create = AsyncMock(return_value=NS(id="resource-id"))
    monkeypatch.setattr(
        number_commands if group == "number" else mailbox_commands,
        "PhoneNumber" if group == "number" else "Mailbox",
        lambda **kwargs: create(**kwargs),
    )
    result = CliRunner().invoke(
        cli, [group, "create", "--yes"] + (["--billing", "paid"] if allowed else [])
    )
    assert result.exit_code == 0
    assert f"Created {group} resource-id" in result.stdout
    assert create.call_args.kwargs["billing"] == "paid"
    assert create.call_args.kwargs["keep"] is True


@pytest.mark.parametrize(
    "group,resource", [("number", PhoneNumber), ("mailbox", Mailbox)]
)
def test_cli_list_and_release(monkeypatch, group, resource):
    module = number_commands if group == "number" else mailbox_commands
    name = "list_numbers" if group == "number" else "list_mailboxes"
    item = NS(
        id="resource-id",
        address=None,
        status="active",
        phone_number=None,
        country_code=None,
        state="active",
    )
    monkeypatch.setattr(module, name, AsyncMock(return_value=[item]))
    release = AsyncMock(
        return_value=NS(data=NS(state="cancel_scheduled", status="cancel_scheduled"))
    )
    monkeypatch.setattr(resource, "release", release)
    assert "resource-id" in CliRunner().invoke(cli, [group, "list"]).output
    assert (
        CliRunner().invoke(cli, [group, "release", "resource-id", "--yes"]).exit_code
        == 0
    )
    release.assert_awaited_once()


@pytest.mark.parametrize(
    "group,command,resource",
    [("number", "code", PhoneNumber), ("mailbox", "code", Mailbox)],
)
def test_codes_are_only_explicit_cli_output(monkeypatch, group, command, resource):
    wait = AsyncMock(return_value="012345")
    monkeypatch.setattr(resource, "wait_for_code", wait)
    result = CliRunner().invoke(
        cli, [group, command, "resource-id", "--sender", "Example"]
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "012345"
    assert "Waiting" in result.stderr
    wait.assert_awaited_once_with(sender="Example", timeout=60)


def test_mailbox_create_included_passes_billing(monkeypatch):
    create = AsyncMock(return_value=NS(id="resource-id"))
    monkeypatch.setattr(mailbox_commands, "Mailbox", lambda **kwargs: create(**kwargs))
    result = CliRunner().invoke(
        cli, ["mailbox", "create", "--yes", "--billing", "included"]
    )
    assert result.exit_code == 0, result.output
    assert create.call_args.kwargs["billing"] == "included"


def test_included_proxy_purchase_rejected_before_api(monkeypatch):
    import mobilerun.cli.proxy.group as proxy_commands

    buy = AsyncMock()
    monkeypatch.setattr(proxy_commands, "buy_proxy", buy)
    result = CliRunner().invoke(
        cli, ["proxy", "buy", "--country", "US", "--billing", "included", "--yes"]
    )
    assert result.exit_code != 0
    assert "require" in result.output.lower()
    buy.assert_not_called()


def test_number_create_included_passes_billing(monkeypatch):
    create = AsyncMock(return_value=NS(id="resource-id"))
    monkeypatch.setattr(
        number_commands, "PhoneNumber", lambda **kwargs: create(**kwargs)
    )
    result = CliRunner().invoke(
        cli, ["number", "create", "--yes", "--billing", "included"]
    )
    assert result.exit_code == 0, result.output
    assert create.call_args.kwargs["billing"] == "included"


@pytest.mark.parametrize("group", ["number", "mailbox"])
def test_create_invalid_billing_is_usage_error(group):
    result = CliRunner().invoke(cli, [group, "create", "--yes", "--billing", "invalid"])
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "group,module,name",
    [
        ("mailbox", mailbox_commands, "Mailbox"),
        ("number", number_commands, "PhoneNumber"),
    ],
)
def test_included_creation_confirmation_has_no_paid_wording(
    monkeypatch, group, module, name, capsys
):
    from mobilerun.cli.base import menu, safety

    create = AsyncMock(return_value=NS(id="resource-id"))
    monkeypatch.setattr(module, name, lambda **kwargs: create(**kwargs))
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(safety, "confirm_prompt_async", AsyncMock(return_value=False))
    result = CliRunner().invoke(cli, [group, "create", "--billing", "included"])
    assert result.exit_code == 0
    assert "included quota only (no charges)" in result.output.lower()
    assert "paid" not in result.output.lower()
    create.assert_not_called()
