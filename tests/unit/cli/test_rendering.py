"""Human-readable and machine-readable command output."""

import importlib
import json
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.base import menu
from mobilerun.cli.base.errors import render_error
from mobilerun.cli.devices.base import selection


def test_render_output_and_streams(monkeypatch):
    from mobilerun.cli.base import render

    @click.command()
    def command():
        render.heading(2, "app")
        render.bullets(["First", "Second"])
        render.kv([("Account", "person")])
        render.success("Done")
        render.warn("Caution")
        render.error("Failed")
        with render.spinner("Working"):
            pass

    monkeypatch.setenv("NO_COLOR", "1")
    result = CliRunner().invoke(command, color=True)
    assert result.exit_code == 0
    assert "Apps · 2" in result.stdout
    assert "  • First" in result.stdout
    assert "Account: person" in result.stdout
    assert "✓ Done" in result.stdout and "! Caution" in result.stdout
    assert "✗ Failed" in result.stderr
    assert "Working" not in result.output
    assert "\x1b[" not in result.output


def test_apps_human_and_json(monkeypatch):

    apps = importlib.import_module("mobilerun.cli.devices.app.list")

    driver = AsyncMock()
    driver.list_apps.return_value = [{"label": "Browser", "package": "com.browser"}]
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=driver))
    runner = CliRunner()
    human = runner.invoke(apps.apps, ["serial", "--android"])
    assert human.exit_code == 0
    assert "Apps · 1" in human.stdout and "Browser" in human.stdout
    assert "com.browser" in human.stdout
    machine = runner.invoke(apps.apps, ["serial", "--android", "--json"])
    assert json.loads(machine.stdout) == [
        {"label": "Browser", "package": "com.browser"}
    ]


def test_press_reports_target(monkeypatch):

    press = importlib.import_module("mobilerun.cli.devices.input.press")

    monkeypatch.setattr(
        selection, "connect_target", AsyncMock(return_value=AsyncMock())
    )
    result = CliRunner().invoke(press.press, ["home", "serial", "--android"])
    assert result.exit_code == 0
    assert (
        "✓" in result.stdout and "serial" in result.stdout and "home" in result.stdout
    )


def test_skills_list_short_long_and_json(monkeypatch):

    from mobilerun.skills.bundle import Skill

    module = importlib.import_module("mobilerun.skills.cli.list")
    monkeypatch.setattr(module, "get_settings", lambda: NS(MOBILERUN_AGENTS={}))
    monkeypatch.setattr(
        module,
        "list_skills",
        lambda: [
            Skill(
                name="sample",
                description="First sentence. Second sentence.",
                path=Path("sample"),
            )
        ],
    )
    runner = CliRunner()
    short = runner.invoke(module.list_command)
    assert "Skills · 1" in short.stdout and "sample" in short.stdout
    assert "Second sentence" not in short.stdout
    assert "Second sentence" in runner.invoke(module.list_command, ["--long"]).stdout
    assert (
        json.loads(runner.invoke(module.list_command, ["--json"]).stdout)[0]["name"]
        == "sample"
    )


def test_macro_list_missing_directory_and_json(tmp_path, monkeypatch):

    command = importlib.import_module("mobilerun.macro.cli.list").list
    missing = str(tmp_path / "missing")
    runner = CliRunner()
    for args in ([missing], [missing, "--json"]):
        result = runner.invoke(command, args)
        assert result.exit_code == 1
        assert f"Macro directory not found: {missing}" in result.output
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(command)
    assert result.exit_code == 0 and "No macros" in result.stdout
    result = runner.invoke(command, ["--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == []


def test_device_action_success_output(monkeypatch, tmp_path):

    from mobilerun.cli.devices import device

    apk = tmp_path / "app.apk"
    apk.write_bytes(b"apk")
    cases = [
        ("tap", ["input", "tap", "1", "2"]),
        ("swipe", ["input", "swipe", "1", "2", "3", "4"]),
        ("long_press", ["input", "long-press", "1", "2"]),
        ("type", ["input", "type", "hello"]),
        ("start", ["app", "start", "app.id"]),
        ("install", ["app", "install", str(apk)]),
        ("uninstall", ["app", "uninstall", "app.id", "--yes"]),
    ]
    for name, args in cases:
        monkeypatch.setattr(
            selection, "connect_target", AsyncMock(return_value=AsyncMock())
        )
        result = CliRunner().invoke(device, [*args, "serial", "--android"])
        assert result.exit_code == 0, (name, result.exception)
        assert "✓" in result.stdout and "serial" in result.stdout, name


def test_pool_failure_lists_each_error_on_stderr():
    from mobilerun.errors.base import DeviceError

    @click.command()
    def command():
        first = DeviceError("disconnected")
        first.add_note("device 0: Fake")
        second = DeviceError("offline")
        second.add_note("device 1: Fake")
        render_error(ExceptionGroup("devices failed", [first, second]))

    result = CliRunner().invoke(command)
    assert not result.stdout
    assert result.stderr.splitlines() == [
        "devices failed",
        "✗ device 0: Fake: disconnected",
        "✗ device 1: Fake: offline",
    ]


def test_proxy_list_and_status_show_metadata_without_credentials(monkeypatch):

    import mobilerun.cli.proxy.group as module

    proxy = NS(
        id="proxy-1", status="active", country="DE", type="mobile", password="secret"
    )
    monkeypatch.setattr(
        module,
        "list_proxies",
        AsyncMock(return_value=NS(items=[proxy], pagination=NS(has_next=False))),
    )
    monkeypatch.setattr(module, "get_proxy", AsyncMock(return_value=proxy))
    runner = CliRunner()
    human = runner.invoke(module.proxy, ["list"])
    assert (
        "Proxies · 1" in human.stdout
        and "DE" in human.stdout
        and "active" in human.stdout
    )
    machine = runner.invoke(module.proxy, ["list", "--json"])
    assert json.loads(machine.stdout)[0]["id"] == "proxy-1"
    assert "secret" not in machine.output
    status = runner.invoke(module.proxy, ["status", "proxy-1"])
    assert "● active" in status.stdout and "DE" in status.stdout


def test_number_and_mailbox_lists_show_metadata_and_json(monkeypatch):

    import mobilerun.cli.mailbox.group as mailbox
    import mobilerun.cli.phone_number.group as number

    monkeypatch.setattr(
        number,
        "list_numbers",
        AsyncMock(
            return_value=[
                NS(id="n1", phone_number="+49123", country_code="DE", state="active")
            ]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        mailbox,
        "list_mailboxes",
        AsyncMock(
            return_value=[NS(id="m1", address="person@example.test", status="active")]
        ),
        raising=False,
    )
    for group, primary in [
        (number.number, "+49123"),
        (mailbox.mailbox, "person@example.test"),
    ]:
        human = CliRunner().invoke(group, ["list"])
        assert (
            human.exit_code == 0
            and primary in human.stdout
            and "active" in human.stdout
        )
        machine = CliRunner().invoke(group, ["list", "--json"])
        assert json.loads(machine.stdout)[0]["id"]


def test_ui_raw_preserves_original_and_default_indents(monkeypatch):

    ui = importlib.import_module("mobilerun.cli.devices.screen.ui")

    driver = AsyncMock()
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=driver))
    monkeypatch.setattr(ui, "log_progress_to_stderr", lambda: None)
    driver._create_state_provider = lambda options: NS(
        get_state=AsyncMock(
            return_value=NS(formatted_text="root\n  child", phone_state=None)
        )
    )
    raw = CliRunner().invoke(ui.ui, ["serial", "--android", "--raw"])
    assert raw.stdout == "root\n  child\n"
    human = CliRunner().invoke(ui.ui, ["serial", "--android"])
    assert human.stdout == "  root\n    child\n"


def test_doctor_result_starts_with_status_symbol(capsys):
    from mobilerun.cli.doctor import _print_result
    from mobilerun.diagnostics.results import CheckResult, Status

    _print_result(CheckResult(name="ADB", status=Status.PASS, message="Ready"))
    assert capsys.readouterr().out.startswith("✓ ADB: Ready")


def test_auth_status_account_fields(monkeypatch):

    module = importlib.import_module("mobilerun.cli.auth.status")
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    monkeypatch.setattr(
        module.credentials, "load_credentials", lambda: {"access_token": "test"}
    )
    monkeypatch.setattr(
        module.deviceauth,
        "fetch_session",
        AsyncMock(
            return_value={"user": {"email": "person@example.test", "id": "user-1"}}
        ),
    )
    result = CliRunner().invoke(module.whoami)
    assert result.exit_code == 0
    assert "Account: person@example.test" in result.stdout
    assert "User ID: user-1" in result.stdout


def test_device_list_json(monkeypatch):

    module = importlib.import_module("mobilerun.cli.devices.list")
    from mobilerun.cli.devices.base import discovery

    monkeypatch.setattr(
        discovery.adb,
        "list",
        AsyncMock(return_value=[NS(serial="local-1", state="device")]),
    )
    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: None)
    result = CliRunner().invoke(module.devices, ["--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == [
        {"id": "local-1", "source": "local", "default": False, "state": "device"}
    ]


def test_spinner_only_uses_tty(monkeypatch):

    from mobilerun.cli.base import render

    class FakeConsole:
        is_terminal = True
        no_color = False

        def status(self, message):
            calls.append(message)
            return nullcontext()

    calls = []
    fake = FakeConsole()
    monkeypatch.setattr(render, "stderr", fake)
    with render.spinner("Working"):
        pass
    assert calls == ["Working"]
    fake.is_terminal = False
    with render.spinner("Hidden"):
        pass
    assert calls == ["Working"]


def test_uninstall_failure_result_is_not_reported_as_success(monkeypatch):

    uninstall = importlib.import_module("mobilerun.cli.devices.app.uninstall")

    driver = AsyncMock()
    driver.uninstall_app.return_value = "Failure [DELETE_FAILED_INTERNAL_ERROR]"
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=driver))
    result = CliRunner().invoke(
        uninstall.uninstall, ["bad.package", "serial", "--android", "--yes"]
    )
    assert result.exit_code != 0
    assert "✓" not in result.stdout
    assert "DELETE_FAILED_INTERNAL_ERROR" in result.stderr


def test_no_color_disables_ansi_on_tty(monkeypatch):
    from io import StringIO

    from rich.console import Console
    from rich.text import Text

    from mobilerun.cli.base import render

    output = StringIO()
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(
        render,
        "stdout",
        Console(file=output, force_terminal=True, color_system="standard"),
    )
    render.bullets([Text("bold name", style="bold red")])
    assert "\x1b[" not in output.getvalue()


def test_device_list_uses_shared_renderer(monkeypatch):

    module = importlib.import_module("mobilerun.cli.devices.list")
    from mobilerun.cli.devices.base import discovery

    monkeypatch.setattr(
        discovery.adb,
        "list",
        AsyncMock(return_value=[NS(serial="serial", state="device")]),
    )
    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: None)
    result = CliRunner().invoke(module.devices)
    assert result.stdout == "Local devices · 1\n\nserial    device\n"


def test_doctor_summary_uses_status_lines(capsys):
    from mobilerun.cli.doctor import _print_summary
    from mobilerun.diagnostics.results import CheckResult, Status

    _print_summary([CheckResult("ADB", Status.WARN, "Unavailable")])
    assert capsys.readouterr().out.splitlines() == ["! 1 warning", "! ADB: Unavailable"]


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


@pytest.mark.parametrize(
    "action,billing,description",
    [
        ("delete file", "no_commerce", "Delete file."),
        ("release mailbox", "paid", "Release mailbox."),
    ],
    ids=["file-delete", "mailbox-release"],
)
def test_non_charge_confirmation_omits_billing(
    monkeypatch, action, billing, description
):
    from mobilerun.cli.base.safety import confirm_action

    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    @click.command()
    def command():
        confirm_action(action, billing, description, yes=True)

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert result.stdout == description + "\n"


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
    pytest.importorskip("mobilerun_agents", reason="requires mobilerun-agents add-on")

    module = importlib.import_module("mobilerun_agents.cli.agent.task")
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
