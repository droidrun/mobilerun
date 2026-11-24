import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

import mobilerun.cli.mailbox.group as mailbox_commands
import mobilerun.cli.phone_number.group as number_commands
from mobilerun.cli.app import cli
from mobilerun.errors import CloudAuthenticationError, CloudPermissionDenied


@pytest.fixture(autouse=True)
def fake_number_client(monkeypatch):
    events = []

    @asynccontextmanager
    async def client(**kwargs):
        events.append("enter")
        try:
            yield object()
        finally:
            events.append("exit")

    monkeypatch.setattr(number_commands, "cloud_client", client)
    return events


@pytest.mark.parametrize(
    "group,module,name",
    [
        ("number", number_commands, "PhoneNumber"),
        ("mailbox", mailbox_commands, "Mailbox"),
    ],
)
def test_create_forwards_plan(monkeypatch, group, module, name):
    create = AsyncMock(return_value=SimpleNamespace(id="id"))
    monkeypatch.setattr(module, name, lambda **kwargs: create(**kwargs))
    client = module.cloud_numbers if group == "number" else module.cloud_mailboxes
    data = SimpleNamespace(
        phone_number=None, address=None, state="active", status="active"
    )
    monkeypatch.setattr(
        client, "retrieve", AsyncMock(return_value=SimpleNamespace(data=data))
    )
    result = CliRunner().invoke(
        cli, [group, "create", "--yes", "--billing", "paid", "--plan", "rent"]
    )
    assert result.exit_code == 0, result.output
    assert create.call_args.kwargs["plan"] == "rent"


@pytest.mark.parametrize(
    "group,module", [("number", number_commands), ("mailbox", mailbox_commands)]
)
def test_rent_included_fails_before_api(monkeypatch, group, module):
    import mobilerun.clients.cloud.mailboxes as mailboxes
    import mobilerun.clients.cloud.numbers as numbers

    sdk = AsyncMock()
    monkeypatch.setattr(
        numbers if group == "number" else mailboxes, "cloud_client", sdk
    )
    result = CliRunner().invoke(
        cli, [group, "create", "--yes", "--billing", "included", "--plan", "rent"]
    )
    assert result.exit_code != 0
    assert "paid" in result.output.lower()
    sdk.assert_not_called()


@pytest.mark.parametrize(
    "group,module,name,args",
    [
        ("number", number_commands, "PhoneNumber", ["--country", "US"]),
        ("mailbox", mailbox_commands, "Mailbox", []),
    ],
)
@pytest.mark.parametrize("as_json", [False, True])
def test_included_command(monkeypatch, group, module, name, args, as_json):
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
        ("number", number_commands, "PhoneNumber", ["--country", "US"]),
        ("mailbox", mailbox_commands, "Mailbox", []),
    ],
)
def test_included_zero_is_not_misrepresented(monkeypatch, group, module, name, args):
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
        SimpleNamespace(country="us", name="United States", in_stock=True),
        SimpleNamespace(country="gb", name="United Kingdom", in_stock=False),
    ]
    monkeypatch.setattr(
        number_commands.cloud_numbers,
        "countries",
        AsyncMock(return_value=SimpleNamespace(data=SimpleNamespace(items=countries))),
    )
    available = AsyncMock(
        side_effect=lambda country, **kwargs: {"us": 3, "gb": 0}[country]
    )
    monkeypatch.setattr(number_commands.PhoneNumber, "included_available", available)
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


def test_number_create_picker_shows_included_counts_and_hides_out_of_stock(monkeypatch):
    countries = [
        SimpleNamespace(country="us", name="United States", in_stock=True),
        SimpleNamespace(country="gb", name="United Kingdom", in_stock=True),
        SimpleNamespace(country="fr", name="France", in_stock=False),
    ]
    monkeypatch.setattr(number_commands, "is_interactive", lambda: True)
    monkeypatch.setattr(
        number_commands.cloud_numbers,
        "countries",
        AsyncMock(return_value=SimpleNamespace(data=SimpleNamespace(items=countries))),
    )
    monkeypatch.setattr(
        number_commands.PhoneNumber,
        "included_available",
        AsyncMock(side_effect=lambda country, **kwargs: {"us": 3, "gb": 0}[country]),
    )
    picker = AsyncMock(return_value=None)
    monkeypatch.setattr(number_commands, "select_prompt", picker)
    result = CliRunner().invoke(cli, ["number", "create", "--yes"])
    assert result.exit_code != 0  # picker cancellation
    choices = picker.call_args.args[1]
    assert [choice.value for choice in choices] == ["us", "gb"]
    assert "United States" in choices[0].label and "3 included left" in choices[0].label
    assert "United Kingdom" in choices[1].label and "0 included" in choices[1].label


@pytest.mark.parametrize("command", ["included", "create"])
def test_country_capacity_partial_failure(monkeypatch, command, fake_number_client):
    countries = [
        SimpleNamespace(country=c, name=n, in_stock=True)
        for c, n in [("us", "United States"), ("gb", "United Kingdom")]
    ]
    monkeypatch.setattr(
        number_commands.cloud_numbers,
        "countries",
        AsyncMock(return_value=SimpleNamespace(data=SimpleNamespace(items=countries))),
    )

    async def capacity(country, **kwargs):
        if country == "gb":
            raise RuntimeError("internal detail")
        return 3

    monkeypatch.setattr(number_commands.PhoneNumber, "included_available", capacity)
    picker = AsyncMock(return_value="gb")
    monkeypatch.setattr(number_commands, "is_interactive", lambda: True)
    monkeypatch.setattr(number_commands, "select_prompt", picker)
    created = []
    if command == "create":

        def phone(**kwargs):
            created.append(kwargs)
            return object()

        phone.included_available = capacity
        monkeypatch.setattr(number_commands, "PhoneNumber", phone)
        ready = SimpleNamespace(id="n1")
        monkeypatch.setattr(
            number_commands, "complete_checkout", AsyncMock(return_value=ready)
        )
        monkeypatch.setattr(
            number_commands.cloud_numbers,
            "retrieve",
            AsyncMock(
                return_value=SimpleNamespace(
                    data=SimpleNamespace(state="active", phone_number="+1555")
                )
            ),
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
    countries = [SimpleNamespace(country="us", name="United States", in_stock=True)]
    monkeypatch.setattr(
        number_commands.cloud_numbers,
        "countries",
        AsyncMock(return_value=SimpleNamespace(data=SimpleNamespace(items=countries))),
    )
    monkeypatch.setattr(
        number_commands.PhoneNumber,
        "included_available",
        AsyncMock(side_effect=error(status=401)),
    )
    monkeypatch.setattr(number_commands, "is_interactive", lambda: True)
    picker = AsyncMock()
    monkeypatch.setattr(number_commands, "select_prompt", picker)
    result = CliRunner().invoke(number_commands.number, [command])
    assert fake_number_client == ["enter", "exit"]
    assert result.exit_code != 0
    assert "HTTP 401" in result.stderr
    assert (
        "login" if error is CloudAuthenticationError else "permission"
    ) in result.stderr
    picker.assert_not_called()
    assert "unavailable for" not in result.stderr


@pytest.mark.parametrize("command", ["included", "create"])
def test_empty_country_inventory(monkeypatch, command):
    monkeypatch.setattr(
        number_commands.cloud_numbers,
        "countries",
        AsyncMock(return_value=SimpleNamespace(data=SimpleNamespace(items=[]))),
    )
    monkeypatch.setattr(number_commands, "is_interactive", lambda: True)
    args = [command] + (["--json"] if command == "included" else [])
    result = CliRunner().invoke(number_commands.number, args)
    if command == "included":
        assert result.exit_code == 0 and json.loads(result.stdout) == []
    else:
        assert "No phone number countries" in result.output


def test_country_capacity_concurrency_is_bounded_and_client_reused(monkeypatch):
    countries = [
        SimpleNamespace(country=str(i), name=str(i), in_stock=True) for i in range(5)
    ]
    monkeypatch.setattr(
        number_commands.cloud_numbers,
        "countries",
        AsyncMock(return_value=SimpleNamespace(data=SimpleNamespace(items=countries))),
    )
    monkeypatch.setattr(
        number_commands,
        "get_settings",
        lambda: SimpleNamespace(MOBILERUN_CLOUD_NUMBER_CAPACITY_CONCURRENCY=2),
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

    monkeypatch.setattr(number_commands.PhoneNumber, "included_available", capacity)
    result = CliRunner().invoke(number_commands.number, ["included", "--json"])
    assert result.exit_code == 0, result.output
    assert peak == 2 and len({id(client) for client in clients}) == 1


def test_country_capacity_failure_human_table(monkeypatch):
    countries = [SimpleNamespace(country="gb", name="United Kingdom", in_stock=True)]
    monkeypatch.setattr(
        number_commands.cloud_numbers,
        "countries",
        AsyncMock(return_value=SimpleNamespace(data=SimpleNamespace(items=countries))),
    )
    monkeypatch.setattr(
        number_commands.PhoneNumber,
        "included_available",
        AsyncMock(side_effect=RuntimeError("private")),
    )
    result = CliRunner().invoke(number_commands.number, ["included"])
    assert result.exit_code == 0
    assert (
        "United Kingdom" in result.stdout
        and "gb" in result.stdout
        and "unavailable" in result.stdout
    )
    assert result.stderr.count("Warning:") == 1 and "private" not in result.stderr
