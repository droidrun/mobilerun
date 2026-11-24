"""Batch cleanup commands process every selected resource."""

import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.cli.proxy import group as proxy


def test_proxy_release_continues_after_failure_and_returns_json_array(monkeypatch):
    cancel = AsyncMock(side_effect=[RuntimeError("secret"), None])
    monkeypatch.setattr(proxy, "cancel_proxy", cancel)
    result = CliRunner().invoke(
        cli, ["proxy", "release", "one", "two", "--yes", "--json"]
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == [
        {
            "id": "one",
            "status": "failed",
            "error": "unexpected failure (use --debug for details)",
        },
        {"id": "two", "status": "released"},
    ]
    assert cancel.await_count == 2


def test_proxy_release_selects_multiple_and_confirms_once(monkeypatch):
    from mobilerun.cli.base import menu, safety
    from mobilerun.cli.base.prompts import SelectChoice

    cancel = AsyncMock()
    confirm = AsyncMock(return_value=True)
    monkeypatch.setattr(proxy, "cancel_proxy", cancel)
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    argument = proxy.proxy.commands["release"].params[0]
    monkeypatch.setattr(
        argument,
        "choices",
        lambda ctx: [SelectChoice("one", "one"), SelectChoice("two", "two")],
    )
    monkeypatch.setattr(menu, "multi_select_prompt", lambda *args: ["one", "two"])
    monkeypatch.setattr(safety, "confirm_prompt_async", confirm)
    result = CliRunner().invoke(cli, ["proxy", "release", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == [
        {"id": "one", "status": "released"},
        {"id": "two", "status": "released"},
    ]
    assert "Release 2 proxies: one, two" in result.stderr
    assert confirm.await_count == 1
    assert cancel.await_count == 2


def test_proxy_release_empty_selection_aborts(monkeypatch):
    from mobilerun.cli.base import menu
    from mobilerun.cli.base.prompts import SelectChoice

    cancel = AsyncMock()
    monkeypatch.setattr(proxy, "cancel_proxy", cancel)
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    argument = proxy.proxy.commands["release"].params[0]
    monkeypatch.setattr(argument, "choices", lambda ctx: [SelectChoice("one", "one")])
    monkeypatch.setattr(menu, "multi_select_prompt", lambda *args: [])
    result = CliRunner().invoke(cli, ["proxy", "release"])
    assert result.exit_code == 0
    assert "Cancelled" in result.output
    cancel.assert_not_awaited()


def test_abort_stops_batch(monkeypatch):
    import click

    cancel = AsyncMock(side_effect=[click.Abort(), None])
    monkeypatch.setattr(proxy, "cancel_proxy", cancel)
    result = CliRunner().invoke(cli, ["proxy", "release", "one", "two", "--yes"])
    assert result.exit_code == 130
    assert cancel.await_count == 1


def test_duplicate_ids_are_processed_once(monkeypatch):
    cancel = AsyncMock()
    monkeypatch.setattr(proxy, "cancel_proxy", cancel)
    result = CliRunner().invoke(
        cli, ["proxy", "release", "one", "two", "one", "--yes", "--json"]
    )
    assert result.exit_code == 0
    assert [item["id"] for item in json.loads(result.stdout)] == ["one", "two"]
    assert cancel.await_count == 2


def test_picker_output_does_not_contaminate_json(monkeypatch):
    from mobilerun.cli.base import menu
    from mobilerun.cli.base.prompts import SelectChoice

    monkeypatch.setattr(proxy, "cancel_proxy", AsyncMock())
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    argument = proxy.proxy.commands["release"].params[0]
    monkeypatch.setattr(argument, "choices", lambda ctx: [SelectChoice("one", "one")])

    def picker(*args):
        print("PICKER")
        return ["one"]

    monkeypatch.setattr(menu, "multi_select_prompt", picker)
    result = CliRunner().invoke(cli, ["proxy", "release", "--yes", "--json"])
    assert json.loads(result.stdout) == [{"id": "one", "status": "released"}]
    assert "PICKER" in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        ("proxy", "release"),
        ("number", "release"),
        ("number", "unbind"),
        ("mailbox", "release"),
        ("workspace", "files", "rm"),
        ("workspace", "files", "remove"),
        ("workspace", "apps", "remove"),
    ],
)
def test_batch_commands_process_in_order_with_billing_and_safe_json(
    monkeypatch, command
):
    from mobilerun.cli.mailbox import group as mailbox
    from mobilerun.cli.phone_number import group as number
    from mobilerun.cli.workspace import group as workspace
    from mobilerun.models.billing import Billing

    calls = []

    async def process(item_id, **kwargs):
        calls.append((item_id, kwargs))
        if item_id in ("one", "device-one"):
            raise RuntimeError("private server detail")
        return NS(data=NS(state="released", status="released"))

    operation = AsyncMock(side_effect=process)
    billing = Billing.NO_COMMERCE
    if command == ("proxy", "release"):
        monkeypatch.setattr(proxy, "cancel_proxy", operation)
    elif command == ("number", "release") or command == ("mailbox", "release"):
        module, name = (
            (number, "PhoneNumber") if command[0] == "number" else (mailbox, "Mailbox")
        )

        def factory(item_id, **kwargs):
            async def release():
                return await operation(item_id, **kwargs)

            return NS(release=release)

        monkeypatch.setattr(module, name, factory)
    elif command == ("number", "unbind"):
        client = AsyncMock()
        client.__aenter__.return_value = client
        monkeypatch.setattr(number, "cloud_client", lambda **kwargs: client)

        async def retrieve(item_id, **kwargs):
            assert kwargs == {"billing": billing, "client": client}
            return NS(data=NS(device_binding=NS(device_id=f"device-{item_id}")))

        monkeypatch.setattr(
            number.cloud_numbers, "retrieve", AsyncMock(side_effect=retrieve)
        )
        monkeypatch.setattr(number.devices, "unbind_number", operation)
    else:
        monkeypatch.setattr(
            workspace.Workspace,
            "delete_file" if command[1] == "files" else "delete_app",
            operation,
        )
    args = [*command, "one", "two", "--yes", "--json"]
    if command[0] != "workspace":
        args += ["--billing", "no_commerce"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1, result.output
    status = (
        "unbound"
        if command == ("number", "unbind")
        else ("deleted" if command[0] == "workspace" else "released")
    )
    assert json.loads(result.stdout) == [
        {
            "id": "one",
            "status": "failed",
            "error": "unexpected failure (use --debug for details)",
        },
        {"id": "two", "status": status},
    ]
    expected_ids = (
        ["device-one", "device-two"]
        if command == ("number", "unbind")
        else ["one", "two"]
    )
    assert [item for item, _ in calls] == expected_ids
    for _, kwargs in calls:
        if command[0] == "workspace":
            assert kwargs == {}
        elif command == ("number", "unbind"):
            assert kwargs == {"billing": billing, "client": client}
        else:
            assert kwargs == {"billing": billing}


def test_number_release_reports_not_cancellable_per_item(monkeypatch):
    from mobilerun.cli.phone_number import group as number
    from mobilerun.errors.cloud import NumberNotCancellable

    async def release(item):
        if item == "one":
            raise NumberNotCancellable(operation="Release number", status=409)
        return NS(data=NS(state="released"))

    monkeypatch.setattr(
        number, "PhoneNumber", lambda item, **kw: NS(release=lambda: release(item))
    )
    result = CliRunner().invoke(
        cli, ["number", "release", "one", "two", "--yes", "--json"]
    )
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert "This number can't be released" in data[0]["error"]
    assert data[1] == {"id": "two", "status": "released"}


def test_click_exit_stops_batch(monkeypatch):
    import click

    cancel = AsyncMock(side_effect=[click.exceptions.Exit(7), None])
    monkeypatch.setattr(proxy, "cancel_proxy", cancel)
    result = CliRunner().invoke(cli, ["proxy", "release", "one", "two", "--yes"])
    assert result.exit_code == 7
    assert cancel.await_count == 1


@pytest.mark.parametrize("resource,noun", [("files", "file"), ("apps", "app")])
def test_workspace_batch_confirmation_uses_full_noun(monkeypatch, resource, noun):
    from mobilerun.cli.base import menu, safety
    from mobilerun.cli.workspace import group as workspace

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(safety, "confirm_prompt_async", AsyncMock(return_value=True))
    monkeypatch.setattr(workspace.Workspace, f"delete_{noun}", AsyncMock())
    command = "rm" if resource == "files" else "remove"
    result = CliRunner().invoke(cli, ["workspace", resource, command, "one", "two"])
    assert result.exit_code == 0
    assert f"Delete 2 workspace {noun}s: one, two." in result.stderr
    assert f"Workspace {noun} one: deleted" in result.stdout


def test_empty_json_picker_cancels_on_stderr(monkeypatch):
    from mobilerun.cli.base import menu
    from mobilerun.cli.base.prompts import SelectChoice

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    argument = proxy.proxy.commands["release"].params[0]
    monkeypatch.setattr(argument, "choices", lambda ctx: [SelectChoice("one", "one")])
    monkeypatch.setattr(menu, "multi_select_prompt", lambda *args: [])
    result = CliRunner().invoke(cli, ["proxy", "release", "--json"])
    assert result.exit_code == 0
    assert result.stdout == ""
    assert "Cancelled." in result.stderr
