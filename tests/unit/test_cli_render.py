import click
import pytest
from click.testing import CliRunner

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
    import importlib
    import json
    from unittest.mock import AsyncMock

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
    import importlib
    from unittest.mock import AsyncMock

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
    import importlib
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from mobilerun.skills.bundle import Skill

    module = importlib.import_module("mobilerun.skills.cli.list")
    monkeypatch.setattr(
        module, "get_settings", lambda: SimpleNamespace(MOBILERUN_AGENTS={})
    )
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
    import importlib
    import json

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
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices import device

    apk = tmp_path / "app.apk"
    apk.write_bytes(b"apk")
    cases = [
        ("tap", ["tap", "1", "2"]),
        ("swipe", ["swipe", "1", "2", "3", "4"]),
        ("long_press", ["long-press", "1", "2"]),
        ("type", ["type", "hello"]),
        ("start", ["start", "app.id"]),
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
    from mobilerun.cli.base.errors import render_error
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
    import json
    from types import SimpleNamespace as NS
    from unittest.mock import AsyncMock

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
    import json
    from types import SimpleNamespace as NS
    from unittest.mock import AsyncMock

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
    import importlib
    from types import SimpleNamespace as NS
    from unittest.mock import AsyncMock

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
    import importlib
    from unittest.mock import AsyncMock

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
    import importlib
    import json
    from types import SimpleNamespace as NS
    from unittest.mock import AsyncMock

    module = importlib.import_module("mobilerun.cli.devices.list")
    from mobilerun.cli.devices.base import discovery

    monkeypatch.setattr(
        discovery.adb, "list", AsyncMock(return_value=[NS(serial="local-1")])
    )
    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: None)
    result = CliRunner().invoke(module.devices, ["--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == [
        {"id": "local-1", "source": "local", "default": False}
    ]


def test_spinner_only_uses_tty(monkeypatch):
    from contextlib import nullcontext

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


def test_oauth_success_uses_shared_style():
    pytest.importorskip("mobilerun.agents", reason="requires mobilerun-agents add-on")
    from mobilerun.agents.cli.base.oauth_output import _print_oauth_login_success

    @click.command()
    def command():
        _print_oauth_login_success("Example", "/somewhere/credential")

    result = CliRunner().invoke(command)
    assert result.stdout.startswith("✓ Example login succeeded.")
    assert "Credentials saved to: /somewhere/credential" in result.stdout


def test_uninstall_failure_result_is_not_reported_as_success(monkeypatch):
    import importlib
    from unittest.mock import AsyncMock

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
    import importlib
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    module = importlib.import_module("mobilerun.cli.devices.list")
    from mobilerun.cli.devices.base import discovery

    monkeypatch.setattr(
        discovery.adb,
        "list",
        AsyncMock(return_value=[SimpleNamespace(serial="serial")]),
    )
    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: None)
    monkeypatch.setattr(module, "resolve_cloud_api_key", lambda: None)
    result = CliRunner().invoke(module.devices)
    assert result.stdout == "Local devices · 1\n\nserial\n"


def test_doctor_summary_uses_status_lines(capsys):
    from mobilerun.cli.doctor import _print_summary
    from mobilerun.diagnostics.results import CheckResult, Status

    _print_summary([CheckResult("ADB", Status.WARN, "Unavailable")])
    assert capsys.readouterr().out.splitlines() == ["! 1 warning", "! ADB: Unavailable"]


def test_llm_wizard_intro_and_summary_use_shared_style(capsys):
    pytest.importorskip("mobilerun.agents", reason="requires mobilerun-agents add-on")
    from mobilerun.agents.cli.agent.llm.wizard import (
        _print_configure_intro,
        _print_configure_summary,
    )

    _print_configure_intro(None)
    _print_configure_summary(
        None,
        family_id="openai",
        auth_mode="api_key",
        provider_label="OpenAI",
        variant_id="openai",
        model="gpt",
        used_advanced_settings=False,
    )
    out = capsys.readouterr().out
    assert "✓ Configuration saved" in out
    assert "Provider: OpenAI" in out and "Model: gpt" in out


def test_llm_login_output_uses_plain_shared_renderer(monkeypatch):
    pytest.importorskip("mobilerun.agents", reason="requires mobilerun-agents add-on")
    import importlib

    from mobilerun.cli.base import render

    module = importlib.import_module("mobilerun.agents.cli.agent.llm.login.openai")
    seen = []
    monkeypatch.setattr(
        module,
        "run_openai_oauth_login",
        lambda *args, **kwargs: seen.append(kwargs["on_output"]),
    )
    monkeypatch.setattr(module, "_print_oauth_login_success", lambda *args: None)
    module._run_openai_oauth_login("/fake", None)
    assert seen == [render.message]
