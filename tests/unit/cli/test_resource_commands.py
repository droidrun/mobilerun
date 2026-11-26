"""CLI contracts for mailbox, phone number and proxy resources."""

import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import click
import pytest
from click.testing import CliRunner

import mobilerun.cli.mailbox.group as mailbox_commands
import mobilerun.cli.phone_number.group as number_commands
import mobilerun.cli.proxy.group as proxy_commands
from mobilerun.cli.app import cli
from mobilerun.cli.mailbox.group import Mailbox
from mobilerun.cli.phone_number.group import PhoneNumber
from mobilerun.models.billing import Billing


@pytest.fixture
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
    "group,module,constructor",
    [
        ("number", number_commands, "PhoneNumber"),
        ("mailbox", mailbox_commands, "Mailbox"),
    ],
    ids=["number", "mailbox"],
)
@pytest.mark.parametrize(
    "billing", [None, "paid", "included"], ids=["default", "paid", "included"]
)
def test_cli_create_billing(
    monkeypatch, fake_resource_retrieval, group, module, constructor, billing
):
    create = AsyncMock(return_value=NS(id="resource-id"))
    monkeypatch.setattr(module, constructor, create)
    args = [group, "create", "--yes"]
    if billing:
        args += ["--billing", billing]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert f"Created {group} resource-id" in result.stdout
    assert create.await_args.kwargs["billing"] == (billing or "paid")
    assert create.await_args.kwargs["keep"] is True


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
def test_included_creation_does_not_prompt(
    monkeypatch, group, module, name, fake_resource_retrieval
):
    from mobilerun.cli.base import menu, safety

    create = AsyncMock(return_value=NS(id="resource-id"))
    monkeypatch.setattr(module, name, lambda **kwargs: create(**kwargs))
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    prompt = AsyncMock(side_effect=AssertionError("included plan never prompts"))
    monkeypatch.setattr(safety, "confirm_prompt_async", prompt)
    args = ["--country", "US"] if group == "number" else []
    result = CliRunner().invoke(
        cli, [group, "create", "--billing", "included", "--plan", "included", *args]
    )
    assert result.exit_code == 0
    prompt.assert_not_awaited()
    assert create.await_args.kwargs["billing"] is Billing.INCLUDED


@pytest.fixture
def cloud(monkeypatch):
    from mobilerun.core.devices.cloud import CloudDevice

    def no_device(*args, **kwargs):
        pytest.fail("binding commands must not construct or close a CloudDevice")

    monkeypatch.setattr(CloudDevice, "__init__", no_device)
    client = AsyncMock()
    client.__aenter__.return_value = client
    monkeypatch.setattr(
        number_commands, "cloud_client", lambda **kwargs: client, raising=False
    )
    from mobilerun.clients.cloud import devices, numbers

    capabilities = AsyncMock(return_value=NS(capabilities=NS(telephony=True)))
    bind = AsyncMock()
    current = AsyncMock(return_value=NS(binding=None))
    unbind = AsyncMock()
    terminate = AsyncMock()
    retrieve = AsyncMock(return_value=NS(data=NS(device_binding=NS(device_id="dev-1"))))
    monkeypatch.setattr(devices, "capabilities", capabilities)
    monkeypatch.setattr(devices, "bind_number", bind)
    monkeypatch.setattr(devices, "number", current)
    monkeypatch.setattr(devices, "unbind_number", unbind)
    monkeypatch.setattr(devices, "terminate", terminate)
    monkeypatch.setattr(numbers, "retrieve", retrieve)
    return NS(
        client=client,
        capabilities=capabilities,
        bind=bind,
        current=current,
        unbind=unbind,
        retrieve=retrieve,
        terminate=terminate,
    )


def test_bind_passes_ids_billing_and_outputs_json(cloud):
    result = CliRunner().invoke(
        cli, ["number", "bind", "num-1", "dev-1", "--billing", "included", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "id": "num-1",
        "status": "bound",
        "number_id": "num-1",
        "device_id": "dev-1",
    }
    cloud.capabilities.assert_awaited_once_with(
        "dev-1", billing=Billing.INCLUDED, client=cloud.client
    )
    cloud.bind.assert_awaited_once_with(
        "dev-1", "num-1", billing=Billing.INCLUDED, client=cloud.client
    )
    cloud.unbind.assert_not_awaited()
    cloud.terminate.assert_not_awaited()


def test_unsupported_device_fails_cleanly(cloud):
    cloud.capabilities.return_value.capabilities.telephony = False
    result = CliRunner().invoke(cli, ["number", "bind", "num-1", "dev-1"])
    assert result.exit_code == 1
    assert "does not support cloud telephony" in result.output
    assert "Traceback" not in result.output
    cloud.bind.assert_not_awaited()


def test_bind_rejects_existing_device_binding(cloud):
    cloud.current.return_value.binding = NS(number_id="other")
    result = CliRunner().invoke(cli, ["number", "bind", "num-1", "dev-1"])
    assert result.exit_code == 1
    assert "already has" in result.output
    cloud.bind.assert_not_awaited()


def test_unbind_resolves_number_and_outputs_json(cloud):
    result = CliRunner().invoke(
        cli,
        ["number", "unbind", "num-1", "--billing", "no_commerce", "--json", "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [{"id": "num-1", "status": "unbound"}]
    cloud.retrieve.assert_awaited_once_with(
        "num-1", billing=Billing.NO_COMMERCE, client=cloud.client
    )
    cloud.unbind.assert_awaited_once_with(
        "dev-1", billing=Billing.NO_COMMERCE, client=cloud.client
    )
    cloud.terminate.assert_not_awaited()


def test_unbind_unbound_number_reports_error(cloud):
    cloud.retrieve.return_value.data.device_binding = None
    result = CliRunner().invoke(cli, ["number", "unbind", "num-1", "--yes"])
    assert result.exit_code == 1
    assert "not bound" in result.output
    cloud.unbind.assert_not_awaited()


def test_commands_are_in_number_menu():
    with click.Context(number_commands.number) as ctx:
        names = {choice.value for choice in number_commands.number.menu_choices(ctx)}
    assert {"bind", "unbind"} <= names


def test_bind_existing_proxy_passes_force_and_closes(monkeypatch):
    phone = NS(close=AsyncMock())
    extension = NS(
        bind=AsyncMock(),
        status=AsyncMock(return_value=NS(connected=True)),
    )
    monkeypatch.setattr(proxy_commands, "CloudDevice", lambda **kwargs: phone)
    monkeypatch.setattr(proxy_commands, "CloudFixedProxy", lambda id: extension)
    result = CliRunner().invoke(
        proxy_commands.proxy, ["bind", "proxy-id", "phone-id", "--force"]
    )
    assert result.exit_code == 0, result.output
    extension.bind.assert_awaited_once_with(phone, force=True)
    phone.close.assert_awaited_once()


def test_proxy_list_paginates_and_status_is_value_safe(monkeypatch):
    listing = AsyncMock(
        side_effect=[
            NS(
                items=[
                    NS(
                        id="one",
                        country=None,
                        status=None,
                        type=None,
                        billing_mode=None,
                    )
                ],
                pagination=NS(has_next=True),
            ),
            NS(
                items=[
                    NS(
                        id="two",
                        country=None,
                        status=None,
                        type=None,
                        billing_mode=None,
                    )
                ],
                pagination=NS(has_next=False),
            ),
        ]
    )
    monkeypatch.setattr(proxy_commands, "list_proxies", listing)
    monkeypatch.setattr(
        proxy_commands,
        "get_proxy",
        AsyncMock(
            return_value=NS(status="active", country=None, password="never-print")
        ),
    )
    result = CliRunner().invoke(cli, ["proxy", "list"])
    assert result.exit_code == 0
    assert [line.rstrip() for line in result.output.splitlines()] == [
        "Proxies · 2",
        "",
        "one        ○ unknown",
        "two        ○ unknown",
    ]
    listing.assert_awaited_with(
        page=2, page_size=100, billing="paid", included_only=False
    )
    result = CliRunner().invoke(cli, ["proxy", "status", "one"])
    assert result.exit_code == 0
    assert "one" in result.output and "● active" in result.output
    assert "never-print" not in result.output


def test_proxy_buy_passes_explicit_billing_and_redacts_credentials(monkeypatch):
    buy = AsyncMock(return_value=NS(id="new", password="never-print"))
    monkeypatch.setattr(proxy_commands, "buy_proxy", buy)
    result = CliRunner().invoke(
        cli, ["proxy", "buy", "--country", "DE", "--billing", "paid", "--yes"]
    )
    assert result.exit_code == 0
    assert result.output == "✓ Bought proxy new (DE)\n"
    buy.assert_awaited_once_with(
        "DE",
        type="residential",
        billing="paid",
        checkout="browser",
        payment_timeout=600,
    )
