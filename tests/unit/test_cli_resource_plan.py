import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
import mobilerun.cli.phone_number.group as number_commands
import mobilerun.cli.mailbox.group as mailbox_commands


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
    resource = module.PhoneNumber if group == "number" else module.Mailbox
    import mobilerun.clients.cloud.numbers as numbers
    import mobilerun.clients.cloud.mailboxes as mailboxes

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
