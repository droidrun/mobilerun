"""Resource pricing, checkout, selection and status at the CLI boundary."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.cli.base import choices, menu
from mobilerun.cli.mailbox import group as mailbox
from mobilerun.cli.phone_number import group as number
from mobilerun.cli.proxy import group as proxy
from mobilerun.errors import CloudAuthenticationError, CloudPermissionDenied
from mobilerun.models.billing import Billing


@pytest.fixture
def fake_number_client(monkeypatch):
    events = []

    @asynccontextmanager
    async def client(**kwargs):
        events.append("enter")
        try:
            yield object()
        finally:
            events.append("exit")

    monkeypatch.setattr(number, "cloud_client", client)
    return events


@pytest.mark.parametrize(
    "group,module,name",
    [
        ("number", number, "PhoneNumber"),
        ("mailbox", mailbox, "Mailbox"),
    ],
)
def test_create_forwards_plan(monkeypatch, group, module, name, fake_number_client):
    create = AsyncMock(return_value=NS(id="id"))
    monkeypatch.setattr(module, name, lambda **kwargs: create(**kwargs))
    client = module.cloud_numbers if group == "number" else module.cloud_mailboxes
    data = NS(phone_number=None, address=None, state="active", status="active")
    monkeypatch.setattr(client, "retrieve", AsyncMock(return_value=NS(data=data)))
    result = CliRunner().invoke(
        cli, [group, "create", "--yes", "--billing", "paid", "--plan", "rent"]
    )
    assert result.exit_code == 0, result.output
    assert create.call_args.kwargs["plan"] == "rent"


@pytest.mark.parametrize(
    "group,module,name,args",
    [
        ("number", number, "PhoneNumber", ["--country", "US"]),
        ("mailbox", mailbox, "Mailbox", []),
    ],
)
@pytest.mark.parametrize("as_json", [False, True])
def test_included_command(
    monkeypatch, group, module, name, args, as_json, fake_number_client
):
    available = AsyncMock(return_value=2)
    monkeypatch.setattr(getattr(module, name), "included_available", available)
    result = CliRunner().invoke(
        cli,
        [
            group,
            "included",
            *args,
            "--billing",
            "no_commerce",
            *(["--json"] if as_json else []),
        ],
    )
    assert result.exit_code == 0, result.output
    if as_json:
        assert json.loads(result.output)["remaining"] == 2
    else:
        assert "Included " in result.output and "2 remaining" in result.output
    available.assert_awaited_once_with(
        *(["US"] if group == "number" else []), billing="no_commerce"
    )


@pytest.mark.parametrize(
    "group,module,name,args",
    [
        ("number", number, "PhoneNumber", ["--country", "US"]),
        ("mailbox", mailbox, "Mailbox", []),
    ],
)
def test_included_zero_is_not_misrepresented(
    monkeypatch, group, module, name, args, fake_number_client
):
    monkeypatch.setattr(
        getattr(module, name), "included_available", AsyncMock(return_value=0)
    )
    result = CliRunner().invoke(cli, [group, "included", *args])
    assert result.exit_code == 0
    assert "0 remaining" in result.output


@pytest.mark.parametrize("as_json", [False, True])
def test_number_included_without_country_lists_all_counts(
    monkeypatch, as_json, fake_number_client
):
    countries = [
        NS(country="us", name="United States", in_stock=True),
        NS(country="gb", name="United Kingdom", in_stock=False),
    ]
    monkeypatch.setattr(
        number.cloud_numbers,
        "countries",
        AsyncMock(return_value=NS(data=NS(items=countries))),
    )
    available = AsyncMock(
        side_effect=lambda country, **kwargs: {"us": 3, "gb": 0}[country]
    )
    monkeypatch.setattr(number.PhoneNumber, "included_available", available)
    result = CliRunner().invoke(
        cli,
        [
            "number",
            "included",
            "--billing",
            "no_commerce",
            *(["--json"] if as_json else []),
        ],
    )
    assert result.exit_code == 0, result.output
    assert fake_number_client == ["enter", "exit"]
    if as_json:
        assert json.loads(result.output) == [
            {"country": "us", "remaining": 3},
            {"country": "gb", "remaining": 0},
        ]
    else:
        assert "United States" in result.output
        assert "us" in result.output and "3 remaining" in result.output
        assert "gb" in result.output and "0 remaining" in result.output


def test_number_create_picker_shows_included_counts_and_hides_out_of_stock(
    monkeypatch, fake_number_client
):
    countries = [
        NS(country="us", name="United States", in_stock=True),
        NS(country="gb", name="United Kingdom", in_stock=True),
        NS(country="fr", name="France", in_stock=False),
    ]
    monkeypatch.setattr(number, "is_interactive", lambda: True)
    monkeypatch.setattr(
        number.cloud_numbers,
        "countries",
        AsyncMock(return_value=NS(data=NS(items=countries))),
    )
    monkeypatch.setattr(
        number.PhoneNumber,
        "included_available",
        AsyncMock(side_effect=lambda country, **kwargs: {"us": 3, "gb": 0}[country]),
    )
    picker = AsyncMock(return_value=None)
    monkeypatch.setattr(number, "select_prompt", picker)
    result = CliRunner().invoke(cli, ["number", "create", "--yes"])
    assert result.exit_code != 0  # picker cancellation
    choices = picker.call_args.args[1]
    assert [choice.value for choice in choices] == ["us", "gb"]
    assert "United States" in choices[0].label and "3 included left" in choices[0].label
    assert "United Kingdom" in choices[1].label and "0 included" in choices[1].label


@pytest.mark.parametrize("command", ["included", "create"])
def test_country_capacity_partial_failure(monkeypatch, command, fake_number_client):
    countries = [
        NS(country=c, name=n, in_stock=True)
        for c, n in [("us", "United States"), ("gb", "United Kingdom")]
    ]
    monkeypatch.setattr(
        number.cloud_numbers,
        "countries",
        AsyncMock(return_value=NS(data=NS(items=countries))),
    )

    async def capacity(country, **kwargs):
        if country == "gb":
            raise RuntimeError("internal detail")
        return 3

    monkeypatch.setattr(number.PhoneNumber, "included_available", capacity)
    picker = AsyncMock(return_value="gb")
    monkeypatch.setattr(number, "is_interactive", lambda: True)
    monkeypatch.setattr(number, "select_prompt", picker)
    created = []
    if command == "create":

        def phone(**kwargs):
            created.append(kwargs)
            return object()

        phone.included_available = capacity
        monkeypatch.setattr(number, "PhoneNumber", phone)
        ready = NS(id="n1")
        monkeypatch.setattr(number, "complete_checkout", AsyncMock(return_value=ready))
        monkeypatch.setattr(
            number.cloud_numbers,
            "retrieve",
            AsyncMock(return_value=NS(data=NS(state="active", phone_number="+1555"))),
        )
    args = ["number", command] + (
        ["--json"] if command == "included" else ["--plan", "rent", "--yes"]
    )
    result = CliRunner().invoke(cli, args)
    assert fake_number_client == ["enter", "exit"]
    assert "gb" in result.stderr and "internal detail" not in result.stderr
    if command == "included":
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == [
            {"country": "us", "remaining": 3},
            {"country": "gb", "remaining": None},
        ]
    else:
        assert result.exit_code == 0, result.output
        assert created[0]["country"] == "gb" and created[0]["plan"] == "rent"
        choices = picker.call_args.args[1]
        assert (
            choices[1].value == "gb"
            and "included count unavailable" in choices[1].label
        )


@pytest.mark.parametrize("error", [CloudAuthenticationError, CloudPermissionDenied])
@pytest.mark.parametrize("command", ["included", "create"])
def test_country_capacity_auth_error_is_not_swallowed(
    monkeypatch, command, error, fake_number_client
):
    countries = [NS(country="us", name="United States", in_stock=True)]
    monkeypatch.setattr(
        number.cloud_numbers,
        "countries",
        AsyncMock(return_value=NS(data=NS(items=countries))),
    )
    monkeypatch.setattr(
        number.PhoneNumber,
        "included_available",
        AsyncMock(side_effect=error(status=401)),
    )
    monkeypatch.setattr(number, "is_interactive", lambda: True)
    picker = AsyncMock()
    monkeypatch.setattr(number, "select_prompt", picker)
    result = CliRunner().invoke(number.number, [command])
    assert fake_number_client == ["enter", "exit"]
    assert result.exit_code != 0
    assert "HTTP 401" in result.stderr
    assert (
        "login" if error is CloudAuthenticationError else "permission"
    ) in result.stderr
    picker.assert_not_called()
    assert "unavailable for" not in result.stderr


@pytest.mark.parametrize("command", ["included", "create"])
def test_empty_country_inventory(monkeypatch, command, fake_number_client):
    monkeypatch.setattr(
        number.cloud_numbers,
        "countries",
        AsyncMock(return_value=NS(data=NS(items=[]))),
    )
    monkeypatch.setattr(number, "is_interactive", lambda: True)
    args = [command] + (["--json"] if command == "included" else [])
    result = CliRunner().invoke(number.number, args)
    if command == "included":
        assert result.exit_code == 0 and json.loads(result.stdout) == []
    else:
        assert "No phone number countries" in result.output


def test_country_capacity_concurrency_is_bounded_and_client_reused(
    monkeypatch, fake_number_client
):
    countries = [NS(country=str(i), name=str(i), in_stock=True) for i in range(5)]
    monkeypatch.setattr(
        number.cloud_numbers,
        "countries",
        AsyncMock(return_value=NS(data=NS(items=countries))),
    )
    monkeypatch.setattr(
        number,
        "get_settings",
        lambda: NS(MOBILERUN_CLOUD_NUMBER_CAPACITY_CONCURRENCY=2),
    )
    active = 0
    peak = 0
    clients = []

    async def capacity(country, *, billing, client):
        nonlocal active, peak
        clients.append(client)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return 1

    monkeypatch.setattr(number.PhoneNumber, "included_available", capacity)
    result = CliRunner().invoke(number.number, ["included", "--json"])
    assert result.exit_code == 0, result.output
    assert peak == 2 and len({id(client) for client in clients}) == 1


def test_country_capacity_failure_human_table(monkeypatch, fake_number_client):
    countries = [NS(country="gb", name="United Kingdom", in_stock=True)]
    monkeypatch.setattr(
        number.cloud_numbers,
        "countries",
        AsyncMock(return_value=NS(data=NS(items=countries))),
    )
    monkeypatch.setattr(
        number.PhoneNumber,
        "included_available",
        AsyncMock(side_effect=RuntimeError("private")),
    )
    result = CliRunner().invoke(number.number, ["included"])
    assert result.exit_code == 0
    assert (
        "United Kingdom" in result.stdout
        and "gb" in result.stdout
        and "unavailable" in result.stdout
    )
    assert result.stderr.count("Warning:") == 1 and "private" not in result.stderr


def test_cloud_devices_order_labels_and_no_commerce(monkeypatch):
    listing = AsyncMock(
        return_value=[
            NS(id="d1", name="Sleeping", state="stopped"),
            NS(id="d2", name="Ready", state="running"),
        ]
    )
    monkeypatch.setattr(choices, "list_devices", listing, raising=False)
    result = choices.cloud_devices(click.Context(click.Command("test")))
    assert [item.value for item in result] == ["d2", "d1"]
    assert "Ready" in result[0].label and "running" in (
        result[0].hint or result[0].label
    )
    assert "stopped" in (result[1].hint or result[1].label)
    listing.assert_awaited_once_with(billing=Billing.NO_COMMERCE, base_url=None)


def test_resource_picker_status_and_order(monkeypatch):
    monkeypatch.setattr(
        choices,
        "list_mailboxes",
        AsyncMock(
            return_value=[
                NS(id="m1", address="a", status="failed"),
                NS(id="m2", address="b", status="active"),
            ]
        ),
    )
    monkeypatch.setattr(
        choices,
        "list_numbers",
        AsyncMock(
            return_value=[
                NS(id="n1", phone_number="1", state="failed"),
                NS(id="n2", phone_number="2", state="active"),
            ]
        ),
    )
    listing = AsyncMock(
        side_effect=[
            NS(
                items=[NS(id="p1", country="us", status="inactive", type="mobile")],
                pagination=NS(has_next=True),
            ),
            NS(
                items=[NS(id="p2", country="gb", status="active", type="residential")],
                pagination=NS(has_next=False),
            ),
        ]
    )
    monkeypatch.setattr(choices, "list_proxies", listing)
    ctx = click.Context(click.Command("test"))
    proxy_choices = choices.proxies(ctx)
    for result, ids, status in [
        (choices.mailboxes(ctx), ["m2", "m1"], "failed"),
        (choices.numbers(ctx), ["n2", "n1"], "failed"),
        (proxy_choices, ["p2", "p1"], "inactive"),
    ]:
        assert [item.value for item in result] == ids
        assert status in (result[1].hint or result[1].label)
    assert "residential" in proxy_choices[0].hint
    assert listing.await_count == 2


def test_proxy_bind_picks_device_and_reports_progress(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(
        choices,
        "list_devices",
        AsyncMock(return_value=[NS(id="dev", name="Phone", state="running")]),
        raising=False,
    )
    seen = []

    def select(label, options):
        seen.append((label, [item.value for item in options]))
        return options[0].value

    monkeypatch.setattr(menu, "select_prompt", select)
    phone = NS(close=AsyncMock())
    extension = NS(bind=AsyncMock(), status=AsyncMock(return_value=NS(connected=True)))
    monkeypatch.setattr(proxy, "CloudDevice", lambda **kwargs: phone)
    monkeypatch.setattr(proxy, "CloudFixedProxy", lambda proxy_id: extension)
    result = CliRunner().invoke(proxy.proxy, ["bind", "proxy"])
    assert result.exit_code == 0, result.output
    assert seen == [("Cloud device ID", ["dev"])]
    assert (
        "Binding proxy" in result.output and "Waiting for connection" in result.output
    )
    assert "✓ Bound proxy proxy to dev" in result.output
    extension.bind.assert_awaited_once_with(phone, force=False)


def test_proxy_bind_conflict_asks_before_retry(monkeypatch):
    from mobilerun.core.proxy import UnmanagedProxyConflict

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    phone = NS(close=AsyncMock())
    extension = NS(
        bind=AsyncMock(
            side_effect=[
                UnmanagedProxyConflict(
                    "Device has an unmanaged active proxy; use force=True to take over"
                ),
                None,
            ]
        ),
        status=AsyncMock(return_value=NS(connected=True)),
    )
    monkeypatch.setattr(proxy, "CloudDevice", lambda **kwargs: phone)
    monkeypatch.setattr(proxy, "CloudFixedProxy", lambda proxy_id: extension)
    monkeypatch.setattr(
        proxy, "confirm_prompt_async", AsyncMock(return_value=True), raising=False
    )
    result = CliRunner().invoke(proxy.proxy, ["bind", "proxy", "dev"])
    assert result.exit_code == 0, result.output
    assert [call.kwargs["force"] for call in extension.bind.await_args_list] == [
        False,
        True,
    ]


def test_proxy_bind_conflict_declined(monkeypatch):
    from mobilerun.core.proxy import UnmanagedProxyConflict

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    phone = NS(close=AsyncMock())
    extension = NS(
        bind=AsyncMock(
            side_effect=UnmanagedProxyConflict(
                "Device has an unmanaged active proxy; use force=True to take over"
            )
        )
    )
    monkeypatch.setattr(proxy, "CloudDevice", lambda **kwargs: phone)
    monkeypatch.setattr(proxy, "CloudFixedProxy", lambda proxy_id: extension)
    monkeypatch.setattr(proxy, "confirm_prompt_async", AsyncMock(return_value=False))
    result = CliRunner().invoke(proxy.proxy, ["bind", "proxy", "dev"])
    assert result.exit_code == 0
    assert "Cancelled" in result.output
    extension.bind.assert_awaited_once_with(phone, force=False)
    phone.close.assert_awaited_once()


def test_proxy_bind_noninteractive_conflict_recommends_force(monkeypatch):
    from mobilerun.core.proxy import UnmanagedProxyConflict

    monkeypatch.setattr(menu, "is_interactive", lambda: False)
    phone = NS(close=AsyncMock())
    extension = NS(
        bind=AsyncMock(
            side_effect=UnmanagedProxyConflict(
                "Device has an unmanaged active proxy; use force=True to take over"
            )
        )
    )
    monkeypatch.setattr(proxy, "CloudDevice", lambda **kwargs: phone)
    monkeypatch.setattr(proxy, "CloudFixedProxy", lambda proxy_id: extension)
    result = CliRunner().invoke(proxy.proxy, ["bind", "proxy", "dev"])
    assert result.exit_code != 0
    assert "--force" in result.output
    assert "force=True" not in result.output


def test_proxy_bind_force_skips_confirmation(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    phone = NS(close=AsyncMock())
    extension = NS(bind=AsyncMock(), status=AsyncMock(return_value=NS(connected=True)))
    prompt = AsyncMock()
    monkeypatch.setattr(proxy, "CloudDevice", lambda **kwargs: phone)
    monkeypatch.setattr(proxy, "CloudFixedProxy", lambda proxy_id: extension)
    monkeypatch.setattr(proxy, "confirm_prompt_async", prompt)
    result = CliRunner().invoke(proxy.proxy, ["bind", "proxy", "dev", "--force"])
    assert result.exit_code == 0
    extension.bind.assert_awaited_once_with(phone, force=True)
    prompt.assert_not_awaited()


def test_proxy_confirmation_default_and_escape(monkeypatch):
    import asyncio

    from mobilerun.cli.base import prompts

    select = AsyncMock(side_effect=["yes", "no", None])
    monkeypatch.setattr(prompts, "select_prompt_async", select)

    async def scenario():
        assert await prompts.confirm_prompt_async("Replace?", default=True)
        assert not await prompts.confirm_prompt_async("Replace?")
        assert not await prompts.confirm_prompt_async("Replace?", default=True)

    asyncio.run(scenario())
    assert [call.kwargs["default"] for call in select.await_args_list] == [
        "yes",
        "no",
        "yes",
    ]


def test_code_timeout_validation():
    runner = CliRunner()
    for command, args in (
        (number.number, ["code", "n1"]),
        (mailbox.mailbox, ["code", "m1"]),
    ):
        result = runner.invoke(command, [*args, "--timeout", "0"])
        assert result.exit_code == 2, result.output
    text = runner.invoke(number.number, ["--help"]).output
    assert "code " in text and "sms " not in text


def test_status_commands_show_resource(monkeypatch):
    monkeypatch.setattr(
        number.cloud_numbers,
        "retrieve",
        AsyncMock(
            return_value=NS(
                data=NS(
                    id="n1",
                    state="active",
                    phone_number="+123",
                    device_binding=NS(device_id="d1"),
                )
            )
        ),
    )
    from mobilerun.clients.cloud import mailboxes as cloud_mailboxes

    monkeypatch.setattr(
        cloud_mailboxes,
        "retrieve",
        AsyncMock(
            return_value=NS(data=NS(id="m1", status="active", address="a@example.com"))
        ),
        raising=False,
    )
    runner = CliRunner()
    n = runner.invoke(number.number, ["status", "n1", "--json"])
    m = runner.invoke(mailbox.mailbox, ["status", "m1", "--json"])
    assert n.exit_code == 0, n.output
    assert json.loads(n.stdout) == {
        "id": "n1",
        "status": "active",
        "phone_number": "+123",
        "device_id": "d1",
    }
    assert m.exit_code == 0, m.output
    assert json.loads(m.stdout) == {
        "id": "m1",
        "status": "active",
        "address": "a@example.com",
    }


def test_code_timeout_is_actionable(monkeypatch):
    for module, group, resource in (
        (number, number.number, "n1"),
        (mailbox, mailbox.mailbox, "m1"),
    ):
        client = number.cloud_numbers if module is number else mailbox.cloud_mailboxes
        monkeypatch.setattr(
            client, "retrieve", AsyncMock(side_effect=RuntimeError("offline"))
        )
        fake = NS(wait_for_code=AsyncMock(side_effect=TimeoutError()))
        monkeypatch.setattr(
            module,
            "PhoneNumber" if module is number else "Mailbox",
            lambda *a, fake=fake, **k: fake,
        )
        result = CliRunner().invoke(
            group, ["code", resource, "--timeout", "2", "--sender", "example"]
        )
        assert result.exit_code != 0
        assert (
            resource in result.output
            and "example" in result.output
            and "2" in result.output
            and "retry" in result.output.lower()
        )
        assert "Waiting" in result.stderr


def test_number_messages_countries_and_send(monkeypatch):
    from pydantic import BaseModel

    class Message(BaseModel):
        body: str

    messages = AsyncMock(return_value=[Message(body="hello")])
    countries = AsyncMock(return_value=NS(data=NS(items=[Message(body="US")])))
    send = AsyncMock()
    monkeypatch.setattr(number.cloud_numbers, "messages", messages)
    monkeypatch.setattr(number.cloud_numbers, "countries", countries)
    monkeypatch.setattr(number.cloud_numbers, "send", send)
    runner = CliRunner()
    result = runner.invoke(number.number, ["messages", "n1", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == [{"body": "hello"}]
    result = runner.invoke(number.number, ["countries", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == [{"body": "US"}]
    result = runner.invoke(
        number.number, ["send", "n1", "+123", "Hi", "--billing", "paid"]
    )
    assert result.exit_code == 0, result.output
    send.assert_awaited_once_with("n1", "+123", "Hi", billing=Billing.PAID)


def test_create_reports_address_and_state(monkeypatch):
    for module, group, name, resource, data in (
        (
            number,
            number.number,
            "PhoneNumber",
            "n1",
            NS(id="n1", state="active", phone_number="+123"),
        ),
        (
            mailbox,
            mailbox.mailbox,
            "Mailbox",
            "m1",
            NS(id="m1", status="active", address="a@example.com"),
        ),
    ):
        monkeypatch.setattr(module, name, lambda *a, **k: object())
        monkeypatch.setattr(
            module, "complete_checkout", AsyncMock(return_value=NS(id=resource))
        )
        client = number.cloud_numbers if module is number else mailbox.cloud_mailboxes
        monkeypatch.setattr(client, "retrieve", AsyncMock(return_value=NS(data=data)))
        result = CliRunner().invoke(group, ["create", "--yes", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["id"] == resource and payload["status"] == "active"
        assert (
            payload.get("phone_number") == "+123"
            or payload.get("address") == "a@example.com"
        )


def test_proxy_json_buy_bind(monkeypatch):
    from mobilerun.cli.base import menu

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    from mobilerun.cli.base import safety

    async def confirm(message, **kwargs):
        print(message)
        return True

    monkeypatch.setattr(safety, "confirm_prompt_async", confirm)

    async def buy(*a, **k):
        print("Complete payment: example")
        return NS(id="p1", status="active")

    monkeypatch.setattr(proxy, "buy_proxy", buy)
    phone = NS(close=AsyncMock())
    extension = NS(bind=AsyncMock(), status=AsyncMock(return_value=NS(connected=True)))
    monkeypatch.setattr(proxy, "CloudDevice", lambda **k: phone)
    monkeypatch.setattr(proxy, "CloudFixedProxy", lambda *a: extension)
    runner = CliRunner()
    result = runner.invoke(
        proxy.proxy,
        ["buy", "--country", "US", "--billing", "paid", "--json"],
        input="y\n",
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["id"] == "p1"
    assert "Continue?" not in result.stderr and "Complete payment" in result.stderr
    result = runner.invoke(proxy.proxy, ["bind", "p1", "d1", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "id": "p1",
        "status": "connected",
        "device_id": "d1",
    }
    assert "Waiting" in result.stderr


def test_create_retrieval_failure_preserves_success(monkeypatch):
    for module, group, name, client in (
        (number, number.number, "PhoneNumber", number.cloud_numbers),
        (mailbox, mailbox.mailbox, "Mailbox", mailbox.cloud_mailboxes),
    ):
        monkeypatch.setattr(module, name, lambda *a, **k: object())
        monkeypatch.setattr(
            module, "complete_checkout", AsyncMock(return_value=NS(id="r1"))
        )
        monkeypatch.setattr(
            client,
            "retrieve",
            AsyncMock(side_effect=RuntimeError("network unavailable")),
        )
        result = CliRunner().invoke(group, ["create", "--yes", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == {"id": "r1", "status": "unknown"}
        assert (
            "r1" in result.stderr
            and "created; details unavailable" in result.stderr
            and "RuntimeError" in result.stderr
        )


def test_timeout_resolves_address_and_falls_back(monkeypatch):
    for module, group, name, client, attr, address in (
        (
            number,
            number.number,
            "PhoneNumber",
            number.cloud_numbers,
            "phone_number",
            "+123",
        ),
        (
            mailbox,
            mailbox.mailbox,
            "Mailbox",
            mailbox.cloud_mailboxes,
            "address",
            "a@example.com",
        ),
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *a, **k: NS(wait_for_code=AsyncMock(side_effect=TimeoutError())),
        )
        monkeypatch.setattr(
            client, "retrieve", AsyncMock(return_value=NS(data=NS(**{attr: address})))
        )
        result = CliRunner().invoke(group, ["code", "r1", "--timeout", "2"])
        assert address in result.output
        monkeypatch.setattr(
            client, "retrieve", AsyncMock(side_effect=RuntimeError("offline"))
        )
        result = CliRunner().invoke(group, ["code", "r1", "--timeout", "2"])
        assert "r1" in result.output and "Retry" in result.output


def test_create_warning_does_not_expose_unexpected_exception_text(monkeypatch):
    monkeypatch.setattr(number, "PhoneNumber", lambda **k: object())
    monkeypatch.setattr(
        number, "complete_checkout", AsyncMock(return_value=NS(id="n1"))
    )
    monkeypatch.setattr(
        number.cloud_numbers,
        "retrieve",
        AsyncMock(side_effect=RuntimeError("secret-token")),
    )
    result = CliRunner().invoke(number.number, ["create", "--yes", "--json"])
    assert result.exit_code == 0
    assert "secret-token" not in result.output
    assert "RuntimeError" in result.stderr


def test_confirmed_number_mailbox_create_json_and_checkout_stderr(monkeypatch):
    from mobilerun.cli.base import menu

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    from mobilerun.cli.base import safety

    async def confirm(message, **kwargs):
        print(message)
        return True

    monkeypatch.setattr(safety, "confirm_prompt_async", confirm)
    for module, group, name, client in (
        (number, number.number, "PhoneNumber", number.cloud_numbers),
        (mailbox, mailbox.mailbox, "Mailbox", mailbox.cloud_mailboxes),
    ):

        async def purchase():
            print("Complete payment: example")
            return NS(id="r1")

        monkeypatch.setattr(module, name, lambda **k: purchase())
        monkeypatch.setattr(
            client,
            "retrieve",
            AsyncMock(
                return_value=NS(
                    data=NS(
                        id="r1",
                        state="active",
                        status="active",
                        phone_number="+123",
                        address="a@example.com",
                    )
                )
            ),
        )
        result = CliRunner().invoke(group, ["create", "--json"], input="y\n")
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["id"] == "r1"
        assert "Continue?" not in result.stderr and "Complete payment" in result.stderr
