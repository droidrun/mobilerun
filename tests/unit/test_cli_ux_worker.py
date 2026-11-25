import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.base import menu
from mobilerun.cli.base.errors import render_error


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


def test_empty_choices_explains(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    @click.command(cls=menu.PromptingCommand)
    @click.argument("device", cls=menu.PromptingArgument, choices=lambda ctx: [])
    def command(device):
        pass

    result = CliRunner().invoke(command, [])
    assert result.exit_code == 0
    assert "No device found." in result.output


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


def test_empty_choices_hint(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    @click.command(cls=menu.PromptingCommand)
    @click.argument(
        "device",
        cls=menu.PromptingArgument,
        choices=lambda ctx: [],
        empty_hint="Create one first.",
    )
    def command(device):
        pass

    result = CliRunner().invoke(command, [])
    assert "No device found. Create one first." in result.output


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
    import importlib
    from pathlib import Path
    from types import SimpleNamespace

    from mobilerun.skills.bundle import Skill

    module = importlib.import_module("mobilerun.skills.cli.list")
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(
            MOBILERUN_AGENTS={"claude": SimpleNamespace(skills=("/home", "/project"))}
        ),
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
