import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner


def test_number_list_plain_and_json(monkeypatch):
    import mobilerun.cli.phone_number.group as module

    monkeypatch.setattr(
        module,
        "list_numbers",
        AsyncMock(
            return_value=[
                NS(
                    id="abcdefgh-1234",
                    phone_number="+49123",
                    country_code="DE",
                    state="active",
                ),
                NS(
                    id="only-identifier",
                    phone_number=None,
                    country_code=None,
                    state="released",
                ),
            ]
        ),
    )
    runner = CliRunner()
    human = runner.invoke(module.number, ["list"], env={"NO_COLOR": "1"})
    assert human.exit_code == 0
    assert [line.rstrip() for line in human.stdout.splitlines()] == [
        "Phone numbers · 2",
        "",
        "+49123             DE    ● active      abcdefgh-1234",
        "only-identifier          ○ released",
    ]
    machine = runner.invoke(module.number, ["list", "--json"])
    assert json.loads(machine.stdout)[0]["id"] == "abcdefgh-1234"


def test_mailbox_empty_is_not_warning(monkeypatch):
    import mobilerun.cli.mailbox.group as module

    monkeypatch.setattr(module, "list_mailboxes", AsyncMock(return_value=[]))
    result = CliRunner().invoke(module.mailbox, ["list"], env={"NO_COLOR": "1"})
    assert result.stdout == "No mailboxes yet.\nCreate one: mobilerun mailbox create\n"


def test_workspace_files_list_plain(monkeypatch):
    import mobilerun.cli.workspace.group as module

    monkeypatch.setattr(
        module.Workspace,
        "list_files",
        AsyncMock(
            return_value=[
                NS(id="abcdefgh-1234", name="photo.png", size_bytes=42),
            ]
        ),
    )
    result = CliRunner().invoke(
        module.workspace, ["files", "list"], env={"NO_COLOR": "1"}
    )
    assert result.stdout == "Files · 1\n\nphoto.png    42    abcdefgh-1234\n"


def test_device_app_list_is_aligned_and_json_unchanged(monkeypatch):
    import importlib

    from mobilerun.cli.devices.base import selection

    module = importlib.import_module("mobilerun.cli.devices.app.list")
    driver = AsyncMock()
    driver.list_apps.return_value = [
        {"label": "Browser", "package": "com.example.browser"}
    ]
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=driver))
    runner = CliRunner()
    result = runner.invoke(module.apps, ["serial", "--android"], env={"NO_COLOR": "1"})
    assert result.stdout == "Apps · 1\n\nBrowser    com.example.browser\n"
    assert (
        json.loads(
            runner.invoke(module.apps, ["serial", "--android", "--json"]).stdout
        )[0]["package"]
        == "com.example.browser"
    )


def test_non_billable_confirmation_does_not_show_billing(monkeypatch):
    import click

    from mobilerun.cli.base import menu
    from mobilerun.cli.base.safety import confirm_action

    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    @click.command()
    def command():
        confirm_action("delete file", "no_commerce", "Delete file.", yes=True)

    result = CliRunner().invoke(command)
    assert result.stdout == "Delete file.\n"


def test_credentials_list_groups_by_app(monkeypatch):
    import mobilerun.cli.credentials.group as module

    records = [
        {
            "package": "com.example.app",
            "name": "personal",
            "fields": ["password", "username"],
            "source": "1password",
        },
        {
            "package": "com.example.app",
            "name": "work",
            "fields": ["username"],
            "source": "vault",
        },
    ]
    monkeypatch.setattr(
        module, "_run", lambda operation, local=False: [] if local else records
    )
    result = CliRunner().invoke(module.credentials, ["list"], env={"NO_COLOR": "1"})
    assert [line.rstrip() for line in result.stdout.splitlines()] == [
        "Credentials · 2",
        "",
        "com.example.app",
        "  personal    username, password    1password",
        "  work        username              vault",
    ]


def test_device_actions_use_table(monkeypatch):
    import importlib

    from mobilerun.cli.devices.base import selection

    module = importlib.import_module("mobilerun.cli.devices.action.list")
    driver = AsyncMock()
    driver.actions = NS(
        list=lambda **kwargs: [
            NS(name="tap", description="Tap screen", input_schema={})
        ]
    )
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=driver))
    result = CliRunner().invoke(
        module.actions, ["serial", "--android"], env={"NO_COLOR": "1"}
    )
    assert result.stdout == "Actions · 1\n\ntap    Tap screen\n"


def test_skills_list_plain(monkeypatch):
    import importlib
    from pathlib import Path

    from mobilerun.skills.bundle import Skill

    module = importlib.import_module("mobilerun.skills.cli.list")
    monkeypatch.setattr(module, "get_settings", lambda: NS(MOBILERUN_AGENTS={}))
    monkeypatch.setattr(
        module,
        "list_skills",
        lambda: [Skill(name="sample", description="Description.", path=Path("sample"))],
    )
    result = CliRunner().invoke(module.list_command, env={"NO_COLOR": "1"})
    assert result.stdout == "Skills · 1\n\nsample    Description.\n"


def test_mailbox_status_uses_status_dot_and_preserves_json(monkeypatch):
    import mobilerun.cli.mailbox.group as module

    monkeypatch.setattr(
        module.cloud_mailboxes,
        "retrieve",
        AsyncMock(
            return_value=NS(
                data=NS(id="mail-1", status="stopped", address="me@example.com")
            )
        ),
    )
    runner = CliRunner()
    result = runner.invoke(module.mailbox, ["status", "mail-1"], env={"NO_COLOR": "1"})
    assert result.stdout == "me@example.com    ○ stopped    mail-1\n"
    assert (
        json.loads(
            runner.invoke(module.mailbox, ["status", "mail-1", "--json"]).stdout
        )["id"]
        == "mail-1"
    )


def test_proxy_status_is_a_status_not_a_success(monkeypatch):
    import mobilerun.cli.proxy.group as module

    monkeypatch.setattr(
        module,
        "get_proxy",
        AsyncMock(
            return_value=NS(id="proxy-1", status="active", country="DE", type="mobile")
        ),
    )
    result = CliRunner().invoke(
        module.proxy, ["status", "proxy-1"], env={"NO_COLOR": "1"}
    )
    assert result.stdout == "proxy-1    DE    ● active    mobile\n"


def test_setting_getters_use_labelled_output(monkeypatch):
    from mobilerun.cli.devices import device
    from mobilerun.cli.devices.base import selection

    driver = AsyncMock()
    driver.get_language.return_value = "de-DE"
    driver.get_timezone.return_value = "Europe/Berlin"
    driver.get_location.return_value = NS(latitude=52.52, longitude=13.405)
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=driver))
    runner = CliRunner()
    for setting, expected in [
        ("language", "Language: de-DE\n"),
        ("timezone", "Timezone: Europe/Berlin\n"),
        ("location", "Location: 52.5200, 13.4050\n"),
    ]:
        result = runner.invoke(
            device,
            ["settings", setting, "get", "serial", "--android"],
            env={"NO_COLOR": "1"},
        )
        assert result.exit_code == 0, result.output
        assert result.stdout == expected
    machine = runner.invoke(
        device, ["settings", "location", "get", "serial", "--android", "--json"]
    )
    assert json.loads(machine.stdout) == {"latitude": 52.52, "longitude": 13.405}


def test_task_status_dot_and_relative_timestamp_preserve_json(monkeypatch):
    pytest.importorskip("mobilerun.agents", reason="requires mobilerun-agents add-on")
    import importlib
    from contextlib import asynccontextmanager
    from datetime import datetime, timedelta, timezone

    module = importlib.import_module("mobilerun.agents.cli.agent.task")
    timestamp = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()
    data = {"status": "running", "created_at": timestamp, "steps": 2}

    @asynccontextmanager
    async def fake_client():
        yield object()

    monkeypatch.setattr(module, "_client", fake_client)
    monkeypatch.setattr(
        module.tasks,
        "status",
        AsyncMock(return_value=NS(model_dump=lambda **kwargs: data)),
    )
    runner = CliRunner()
    result = runner.invoke(module.task, ["status", "task-1"], env={"NO_COLOR": "1"})
    assert result.exit_code == 0, result.output
    assert "Status: ● running" in result.stdout
    assert "Created at: 3 min ago" in result.stdout
    assert (
        json.loads(runner.invoke(module.task, ["status", "task-1", "--json"]).stdout)[
            "created_at"
        ]
        == timestamp
    )


def test_non_charge_release_confirmation_omits_billing(monkeypatch):
    import click

    from mobilerun.cli.base import menu
    from mobilerun.cli.base.safety import confirm_action

    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    @click.command()
    def command():
        confirm_action("release mailbox", "paid", "Release mailbox.", yes=True)

    assert CliRunner().invoke(command).stdout == "Release mailbox.\n"


def test_included_resources_are_labelled_and_json_unchanged(monkeypatch):
    import mobilerun.cli.mailbox.group as mailbox
    import mobilerun.cli.phone_number.group as number

    monkeypatch.setattr(
        number.PhoneNumber, "included_available", AsyncMock(return_value=2)
    )
    monkeypatch.setattr(
        mailbox.Mailbox, "included_available", AsyncMock(return_value=3)
    )
    runner = CliRunner()
    assert (
        runner.invoke(number.number, ["included", "--country", "US"]).stdout
        == "Included numbers (US): 2 remaining\n"
    )
    assert json.loads(
        runner.invoke(number.number, ["included", "--country", "US", "--json"]).stdout
    ) == {"country": "US", "remaining": 2}
    assert (
        runner.invoke(mailbox.mailbox, ["included"]).stdout
        == "Included mailboxes: 3 remaining\n"
    )
    assert json.loads(
        runner.invoke(mailbox.mailbox, ["included", "--json"]).stdout
    ) == {"remaining": 3}


def test_cloud_device_status_uses_name_and_state(monkeypatch):
    import importlib
    from contextlib import asynccontextmanager

    module = importlib.import_module("mobilerun.cli.devices.cloud.status")

    @asynccontextmanager
    async def client(**kwargs):
        yield object()

    monkeypatch.setattr(module, "cloud_client", client)
    monkeypatch.setattr(
        module.devices,
        "retrieve",
        AsyncMock(
            return_value=NS(
                name="Tester 1",
                state="ready",
                model_dump_json=lambda: '{"state":"ready"}',
            )
        ),
    )
    result = CliRunner().invoke(
        module.status, ["11111111-1111-1111-1111-111111111111"], env={"NO_COLOR": "1"}
    )
    assert result.exit_code == 0, result.output
    assert (
        result.stdout == "Tester 1    ● ready    11111111-1111-1111-1111-111111111111\n"
    )


def test_device_action_acknowledgements(monkeypatch):
    from mobilerun.cli.devices import device
    from mobilerun.cli.devices.base import selection

    driver = AsyncMock()
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=driver))
    runner = CliRunner()
    for args, expected in [
        (
            ["settings", "timezone", "set", "Europe/Berlin"],
            "✓ Timezone set to Europe/Berlin on serial\n",
        ),
        (["settings", "location", "reset"], "✓ Location reset on serial\n"),
        (["app", "stop", "com.example.app"], "✓ Stopped com.example.app on serial\n"),
        (["file", "rm", "/sdcard/a", "--yes"], "✓ Deleted /sdcard/a from serial\n"),
    ]:
        result = runner.invoke(
            device, [*args, "serial", "--android"], env={"NO_COLOR": "1"}
        )
        assert result.exit_code == 0, (result.output, result.exception)
        assert result.stdout == expected


def test_workspace_delete_acknowledges_and_preserves_json(monkeypatch):
    import mobilerun.cli.workspace.group as module

    monkeypatch.setattr(module.Workspace, "delete_file", AsyncMock())
    runner = CliRunner()
    result = runner.invoke(
        module.workspace, ["files", "rm", "file-1", "--yes"], env={"NO_COLOR": "1"}
    )
    assert result.stdout == "✓ Workspace file file-1: deleted\n"
    assert json.loads(
        runner.invoke(
            module.workspace, ["files", "rm", "file-1", "--yes", "--json"]
        ).stdout
    ) == [{"id": "file-1", "status": "deleted"}]
