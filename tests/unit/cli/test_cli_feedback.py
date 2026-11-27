"""Error reporting, usage help, and interactive command feedback."""

import importlib
from pathlib import Path
from types import SimpleNamespace as NS

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.app import LazyGroup
from mobilerun.cli.base import menu
from mobilerun.cli.base.errors import render_error
from mobilerun.errors.base import DeviceError


class SampleGroup(LazyGroup):
    list_commands = click.Group.list_commands
    get_command = click.Group.get_command


@click.group(cls=SampleGroup)
@click.option("--debug", is_flag=True)
def sample(debug):
    pass


@sample.command()
def fail():
    raise DeviceError("bad\nrequest\x1b[31m")


@sample.command()
def bug():
    raise RuntimeError("internal secret")


@sample.command()
def interrupt():
    raise KeyboardInterrupt()


@sample.command()
def group_failure():
    first = DeviceError("one")
    first.add_note("device 0: Fake")
    raise ExceptionGroup("devices failed", [first, DeviceError("two")])


@sample.command()
def click_failure():
    raise click.ClickException("first\nsecond")


def test_expected_error_is_one_line():
    result = CliRunner().invoke(sample, ["fail"], color=True)
    assert result.exit_code == 1
    assert len(result.output.splitlines()) == 1
    assert "bad request" in result.output
    assert "\x1b" not in result.output


def test_unexpected_error_hides_details_unless_debug():
    result = CliRunner().invoke(sample, ["bug"])
    assert result.exit_code == 1
    assert "internal secret" not in result.output
    assert "--debug" in result.output
    debug = CliRunner().invoke(sample, ["--debug", "bug"])
    assert debug.exit_code == 1
    assert "Traceback" in debug.output and "internal secret" in debug.output


def test_interrupt_and_usage_codes():
    runner = CliRunner()
    assert runner.invoke(sample, ["interrupt"]).exit_code == 130
    assert runner.invoke(sample, ["missing"]).exit_code == 2


def test_group_reports_each_cause():
    result = CliRunner().invoke(sample, ["group-failure"])
    assert result.exit_code == 1
    assert len(result.output.splitlines()) == 3
    assert "device 0" in result.output


def test_click_failure_is_one_line():
    result = CliRunner().invoke(sample, ["click-failure"])
    assert result.exit_code == 1
    assert result.output.splitlines() == ["Error: first second"]


def test_exception_group_with_auth_cause_renders_only_auth_message(capsys):
    from mobilerun.errors.cloud import CloudAuthRequired

    error = ExceptionGroup("group", [RuntimeError("private")])
    error.__cause__ = CloudAuthRequired("not logged in")
    render_error(error)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Error: Not logged in. Run `mobilerun account login` or set MOBILERUN_API_KEY.\n"
    )


def test_variadic_prompt_collects_entries(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    answers = iter(["one", "two", ""])
    monkeypatch.setattr(menu, "text_prompt", lambda label: next(answers))

    @click.command(cls=menu.PromptingCommand)
    @click.argument("files", nargs=-1, required=True)
    def command(files):
        click.echo(repr(files))

    result = CliRunner().invoke(command, [])
    assert result.exit_code == 0
    assert "('one', 'two')" in result.output


def test_provider_failure_falls_back(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "text_prompt", lambda label: "manual")

    def fail(ctx):
        raise RuntimeError("offline")

    @click.command(cls=menu.PromptingCommand)
    @click.argument("device", cls=menu.PromptingArgument, choices=fail)
    def command(device):
        click.echo(device)

    result = CliRunner().invoke(command, [])
    assert result.exit_code == 0
    assert "Warning" in result.output and "manual" in result.output


def test_explicit_short_help_is_not_truncated():
    @click.group(cls=menu.MenuGroup)
    def group():
        pass

    @group.command(short_help="A deliberately long explicit short label for a command")
    def command():
        pass

    with click.Context(group) as ctx:
        assert (
            group.menu_choices(ctx)[0].hint
            == "A deliberately long explicit short label for a command"
        )


def test_timeout_message():
    @click.command()
    def command():
        render_error(TimeoutError("Timed out after 30s"))

    assert "Timed out after 30s" in CliRunner().invoke(command).output


@pytest.mark.parametrize(
    ("module_name", "extra"),
    [
        ("anthropic", "anthropic"),
        ("langfuse", "langfuse"),
        ("openinference", "langfuse"),
    ],
)
def test_known_optional_dependency_message(module_name, extra):
    @click.command()
    def command():
        render_error(
            ModuleNotFoundError(f"No module named '{module_name}'", name=module_name)
        )

    assert (
        f"pip install 'mobilerun-agents[{extra}]'" in CliRunner().invoke(command).output
    )


def test_account_canonical_and_auth_alias():
    from mobilerun.cli.app import cli

    runner = CliRunner()
    assert "account" in runner.invoke(cli, ["--help"]).output
    assert runner.invoke(cli, ["account", "--help"]).exit_code == 0
    assert runner.invoke(cli, ["sandbox", "--help"]).exit_code == 0


def test_variadic_choices_use_multiselect(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(
        menu, "multi_select_prompt", lambda label, choices: ["one", "two"]
    )

    @click.command(cls=menu.PromptingCommand)
    @click.argument(
        "items",
        nargs=-1,
        required=True,
        cls=menu.PromptingArgument,
        choices=lambda ctx: [
            menu.SelectChoice("one", "One"),
            menu.SelectChoice("two", "Two"),
        ],
    )
    def command(items):
        click.echo(repr(items))

    result = CliRunner().invoke(command, [])
    assert result.exit_code == 0
    assert "('one', 'two')" in result.output


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, "No device found."),
        ({"empty_hint": "Create one first."}, "No device found. Create one first."),
    ],
    ids=["default", "hint"],
)
def test_empty_choices_message(monkeypatch, options, expected):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    @click.command(cls=menu.PromptingCommand)
    @click.argument(
        "device",
        cls=menu.PromptingArgument,
        choices=lambda ctx: [],
        **options,
    )
    def command(device):
        pass

    result = CliRunner().invoke(command, [])
    assert result.exit_code == 0, result.output
    assert result.stderr.strip() == expected


def test_missing_credentials_message():
    from mobilerun.errors.cloud import CloudAuthRequired

    @click.command()
    def command():
        render_error(CloudAuthRequired(code="credentials_required"))

    assert (
        "Not logged in. Run `mobilerun account login` or set MOBILERUN_API_KEY."
        in CliRunner().invoke(command).output
    )


def test_skill_list_shows_installed_targets(monkeypatch):

    from mobilerun.skills.bundle import Skill

    module = importlib.import_module("mobilerun.skills.cli.list")
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: NS(MOBILERUN_AGENTS={"claude": NS(skills=("/home", "/project"))}),
    )
    monkeypatch.setattr(
        module,
        "installed",
        lambda agent, scope: {"sample"} if scope == "user" else set(),
    )
    monkeypatch.setattr(
        module,
        "list_skills",
        lambda: [
            Skill(
                name="sample",
                description="Example skill",
                path=Path("sample"),
                relative_path=Path("sample"),
            )
        ],
    )
    result = CliRunner().invoke(module.list_command, [])
    assert result.exit_code == 0, result.output
    assert "claude:user" in result.output


def test_unknown_missing_module_keeps_debug_hint():
    @click.command()
    def command():
        render_error(ModuleNotFoundError("missing", name="mobilerun.internal"))

    assert "--debug" in CliRunner().invoke(command).output


def test_variadic_picker_cancel_is_clean(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "multi_select_prompt", lambda label, choices: None)

    @click.command(cls=menu.PromptingCommand)
    @click.argument(
        "items",
        nargs=-1,
        required=True,
        cls=menu.PromptingArgument,
        choices=lambda ctx: [menu.SelectChoice("one", "One")],
    )
    def command(items):
        click.echo("unexpected")

    result = CliRunner().invoke(command, [])
    assert result.exit_code == 0
    assert "unexpected" not in result.output


def test_macro_missing_explicit_and_implicit_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mobilerun.macro.cli import macro

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    missing = runner.invoke(macro, ["list", "missing"])
    assert missing.exit_code == 1
    assert "Macro directory not found: missing" in missing.output
    default = runner.invoke(macro, ["list"])
    assert default.exit_code == 0
    assert "No macros yet" in default.output
    as_json = runner.invoke(macro, ["list", "--json"])
    assert as_json.exit_code == 0
    assert as_json.output.strip() == "[]"


def test_unknown_skills_offer_list_and_close_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mobilerun.skills import bundle
    from mobilerun.skills.cli.group import skills

    source = tmp_path / "source"
    (source / "alpha").mkdir(parents=True)
    (source / "alpha" / "SKILL.md").write_text("---\ndescription: alpha\n---\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: source)
    runner = CliRunner()
    for args in (["show", "alpa"], ["install", "alpa", "--agent", "codex"]):
        result = runner.invoke(skills, args)
        assert result.exit_code == 1
        assert (
            "Unknown skill: alpa. Run `mobilerun skills list` to see available skills."
            in result.output
        )
        assert "Did you mean 'alpha'?" in result.output
    result = runner.invoke(skills, ["show", "bogus"])
    assert result.exit_code == 1
    assert "Did you mean" not in result.output


def test_sandbox_visible_in_root_help() -> None:
    from mobilerun.cli.app import cli

    output = CliRunner().invoke(cli, ["--help"]).output
    assert "sandbox" in output and "Deploy and run cloud functions." in output


def test_sandbox_visible_in_root_menu() -> None:

    from mobilerun.cli.app import cli

    with click.Context(cli) as ctx:
        choice = next(
            choice for choice in cli.menu_choices(ctx) if choice.value == "sandbox"
        )
        assert choice.hint == "Deploy and run cloud functions."
