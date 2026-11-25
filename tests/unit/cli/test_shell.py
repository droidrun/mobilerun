"""CLI startup, logging, menus, and shell completion contracts."""

import importlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.cli.base import menu

completion = importlib.import_module("mobilerun.cli.base.completion")


def test_detected_zsh_installs_once_and_does_not_dump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setattr(completion, "_stdout_is_tty", lambda: True, raising=False)
    for _ in range(2):
        result = CliRunner().invoke(completion.completion, ["--yes"])
        assert result.exit_code == 0, result.output
        assert "Installed zsh completion" in result.output
        assert "_MOBILERUN_COMPLETE" not in result.output
    script = tmp_path / ".mobilerun/completion.zsh"
    assert "_MOBILERUN_COMPLETE" in script.read_text()
    assert (tmp_path / ".zshrc").read_text().count("source ") == 1


def test_decline_leaves_rc_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setattr(completion, "_stdout_is_tty", lambda: True, raising=False)
    rc = tmp_path / ".zshrc"
    rc.write_text("# existing\n")
    result = CliRunner().invoke(completion.completion, [], input="n\n")
    assert result.exit_code == 0, result.output
    assert rc.read_text() == "# existing\n"
    assert (tmp_path / ".mobilerun/completion.zsh").exists()


@pytest.mark.parametrize(
    "tty,options", [(False, []), (True, ["--print"])], ids=["non-tty", "explicit-print"]
)
def test_completion_prints_script_without_installing(
    tmp_path, monkeypatch, tty, options
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setattr(completion, "_stdout_is_tty", lambda: tty)
    result = CliRunner().invoke(completion.completion, ["zsh", *options])
    assert result.exit_code == 0, result.output
    assert "_MOBILERUN_COMPLETE" in result.output
    assert not (tmp_path / ".mobilerun").exists()


def test_old_bash_reports_clear_error_without_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(completion, "_bash_version", lambda: (3, 2), raising=False)
    result = CliRunner().invoke(completion.completion, ["bash", "--print"])
    assert result.exit_code == 1
    assert "bash 3.2 is too old for completion (needs 4.4+)" in result.output
    assert "Shell completion is not supported" not in result.output
    assert "_MOBILERUN_COMPLETE" not in result.output


def test_fish_installs_to_autoload_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/opt/homebrew/bin/fish")
    monkeypatch.setattr(completion, "_stdout_is_tty", lambda: True, raising=False)
    result = CliRunner().invoke(completion.completion, [])
    assert result.exit_code == 0, result.output
    assert (
        "_MOBILERUN_COMPLETE"
        in (tmp_path / ".config/fish/completions/mobilerun.fish").read_text()
    )
    assert not (tmp_path / ".config/fish/config.fish").exists()


def test_unknown_shell_interactively_asks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/unknown")
    monkeypatch.setattr(completion, "_stdout_is_tty", lambda: True, raising=False)
    monkeypatch.setattr(
        completion, "select_prompt", lambda message, choices: "zsh", raising=False
    )
    result = CliRunner().invoke(completion.completion, ["--yes"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".mobilerun/completion.zsh").exists()


def test_root_menu_completion_installs_instead_of_printing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setattr(completion, "_stdout_is_tty", lambda: True)
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "select_prompt", lambda message, choices: "completion")
    monkeypatch.setattr(completion, "confirm_prompt", lambda message, **kwargs: True)
    result = CliRunner().invoke(cli, [])
    assert result.exit_code == 0, result.output
    assert "Installed zsh completion" in result.output
    assert "_MOBILERUN_COMPLETE" not in result.output
    assert (tmp_path / ".zshrc").exists()


def test_bash_version_uses_selected_bash(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("SHELL", "/opt/homebrew/bin/bash")
    calls: list[list[str]] = []

    def run(args: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(args)
        return SimpleNamespace(stdout="5.2.37(1)-release")

    monkeypatch.setattr(completion.subprocess, "run", run)
    assert completion._bash_version() == (5, 2)
    assert calls[0][0] == "/opt/homebrew/bin/bash"


def test_non_tty_unknown_shell_never_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/bin/unknown")
    monkeypatch.setattr(completion, "_stdout_is_tty", lambda: False)
    monkeypatch.setattr(
        completion,
        "select_prompt",
        lambda *args: pytest.fail("must not prompt in non-interactive mode"),
    )
    result = CliRunner().invoke(completion.completion, [])
    assert result.exit_code == 1
    assert "Cannot detect shell" in result.output
    assert result.stdout == ""


def test_interactive_completion_decline_uses_default_no(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setattr(completion, "_stdout_is_tty", lambda: True)
    monkeypatch.setattr(completion.menu, "is_interactive", lambda: True)
    observed = []
    monkeypatch.setattr(
        completion,
        "confirm_prompt",
        lambda message, **kw: observed.append((message, kw)) or False,
    )
    result = CliRunner().invoke(completion.completion, [])
    assert result.exit_code == 0
    assert observed == [("Add completion to ~/.zshrc?", {"default": False})]
    assert not (tmp_path / ".zshrc").exists()


def distribution(path: Path, name: str, entry: str) -> None:
    """Expose real metadata to importlib without installing a package."""
    info = path / f"{name}-1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Name: {name}\nVersion: 1\n")
    (info / "entry_points.txt").write_text(f"[mobilerun.cli]\n{entry}\n")


def test_malformed_distribution_does_not_disable_healthy_plugins(tmp_path, monkeypatch):
    distribution(tmp_path, "invalidmetadata", "malformed-line")
    distribution(tmp_path, "validmetadata", "healthy = mobilerun.cli.doctor:doctor")
    monkeypatch.syspath_prepend(str(tmp_path))
    runner = CliRunner()
    help_result = runner.invoke(cli, ["--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "healthy" in help_result.stdout and "device" in help_result.stdout
    assert "warning" in help_result.stderr.lower()
    assert "malformed-line" not in help_result.stderr
    assert runner.invoke(cli, ["healthy", "--help"]).exit_code == 0


def test_plugins_list_skips_malformed_distribution(tmp_path, monkeypatch):
    distribution(tmp_path, "invalidmetadata", "malformed-line")
    distribution(tmp_path, "validmetadata", "healthy = mobilerun.cli.doctor:doctor")
    monkeypatch.syspath_prepend(str(tmp_path))
    result = CliRunner().invoke(cli, ["plugins", "list", "--json"])
    assert result.exit_code == 0, result.output
    assert "warning" in result.stderr.lower()
    assert "healthy" in {row["name"] for row in json.loads(result.stdout)}


def test_root_completion_ignores_broken_plugin_target(tmp_path, monkeypatch):
    distribution(tmp_path, "brokencompletion", "broken = nonexistent_target:command")
    monkeypatch.syspath_prepend(str(tmp_path))
    ctx = click.Context(cli)
    assert "device" in {item.value for item in cli.shell_complete(ctx, "")}
    assert "--debug" in {item.value for item in cli.shell_complete(ctx, "--d")}


def test_import_main_returns_without_running_cli():
    result = subprocess.run(
        [sys.executable, "-c", "import mobilerun.__main__; print('imported')"],
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "imported"


@pytest.mark.parametrize("command", [["account"], ["account", "logout"]])
def test_account_help_with_invalid_unrelated_setting(command, monkeypatch):
    monkeypatch.setenv("MOBILERUN_PORTAL_REMOTE_PORT", "invalid")
    result = subprocess.run(
        [sys.executable, "-m", "mobilerun", *command, "--help"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout


def test_repeated_cli_invocations_emit_each_log_message_once(monkeypatch, capsys):
    logger = logging.getLogger("mobilerun")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logger.level)
    monkeypatch.setattr(logger, "propagate", logger.propagate)
    runner = CliRunner()
    for _ in range(2):
        result = runner.invoke(cli, ["device", "--help"])
        assert result.exit_code == 0, result.output
    logger.info("shell logging sentinel")
    assert capsys.readouterr().err.splitlines().count("shell logging sentinel") == 1
    assert len(logger.handlers) == 1


def test_root_debug_enables_debug_logging():

    logger = logging.getLogger("mobilerun")
    old_level = logger.level
    old_handlers = logger.handlers[:]
    old_propagate = logger.propagate
    try:
        result = CliRunner().invoke(cli, ["--debug", "device", "--help"])
        assert result.exit_code == 0
        assert logger.level == logging.DEBUG
    finally:
        logger.setLevel(old_level)
        logger.handlers = old_handlers
        logger.propagate = old_propagate


def test_cli_log_handler_writes_to_stderr():
    from mobilerun.logs.handlers import CLILogHandler

    @click.command()
    def command():
        handler = CLILogHandler()
        handler.emit(
            logging.LogRecord("mobilerun", logging.INFO, "", 1, "diagnostic", (), None)
        )

    result = CliRunner().invoke(command)
    assert result.stdout == ""
    assert "diagnostic" in result.stderr


def test_ping_missing_device_fails_on_stderr(monkeypatch):
    module = importlib.import_module("mobilerun.cli.devices.ping")

    async def missing(target):
        return None

    monkeypatch.setattr(module.adb, "device", missing)
    result = CliRunner().invoke(module.ping, ["missing-device"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "missing-device" in result.stderr


def test_ping_success_names_device(monkeypatch):
    module = importlib.import_module("mobilerun.cli.devices.ping")

    async def device(target):
        return object()

    async def ping(*args):
        pass

    monkeypatch.setattr(module.adb, "device", device)
    monkeypatch.setattr(module, "ping_portal", ping)
    monkeypatch.setattr(module, "ping_portal_content", ping)
    monkeypatch.setattr(module, "ping_portal_tcp", ping)
    result = CliRunner().invoke(module.ping, ["test-device"])
    assert result.exit_code == 0
    assert "✓" in result.stdout
    assert "test-device" in result.stdout


def test_interactive_group_invokes_selection(monkeypatch):

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "select_prompt", lambda message, choices: "hello")

    @click.group(cls=menu.MenuGroup)
    def group():
        pass

    @group.command()
    def hello():
        click.echo("selected")

    result = CliRunner().invoke(group, [])
    assert result.exit_code == 0
    assert "selected" in result.output


def test_interactive_group_prints_epilog_before_prompt(monkeypatch):

    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    def select(message, choices):
        click.echo("PROMPT")
        return None

    monkeypatch.setattr(menu, "select_prompt", select)

    @click.group(cls=menu.MenuGroup, epilog="A helpful tip")
    def group():
        pass

    result = CliRunner().invoke(group, [])
    assert result.exit_code == 0
    assert result.output.index("A helpful tip") < result.output.index("PROMPT")


def test_interactive_group_without_epilog_prints_no_extra_hint(monkeypatch):

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "select_prompt", lambda message, choices: None)

    @click.group(cls=menu.MenuGroup)
    def group():
        pass

    result = CliRunner().invoke(group, [])
    assert result.exit_code == 0
    assert "Usage:" not in result.output


def test_noninteractive_group_still_shows_help():
    from mobilerun.cli.base.menu import MenuGroup

    @click.group(cls=MenuGroup)
    def group():
        pass

    @group.command()
    def hello():
        pass

    result = CliRunner().invoke(group, [])
    assert result.exit_code == 2
    assert "Usage:" in result.output


def test_workspace_apps_help_mentions_dashboard_app_store():
    from mobilerun.cli.workspace.group import workspace

    result = CliRunner().invoke(workspace, ["apps", "--help"])
    assert result.exit_code == 0
    assert "https://cloud.mobilerun.ai/apps" in result.output


def test_root_menu_does_not_load_commands_until_selected(monkeypatch):
    from mobilerun.cli import app

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "select_prompt", lambda message, choices: None)
    monkeypatch.setattr(
        app, "import_module", lambda name: (_ for _ in ()).throw(AssertionError(name))
    )
    result = CliRunner().invoke(app.cli, [])
    assert result.exit_code == 0


def test_keyboard_interrupt_exits_130(monkeypatch):

    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    def interrupt(message, choices):
        raise KeyboardInterrupt

    monkeypatch.setattr(menu, "select_prompt", interrupt)

    @click.group(cls=menu.MenuGroup)
    def group():
        pass

    assert CliRunner().invoke(group, []).exit_code == 130


def test_required_choice_argument_is_selected_in_terminal(monkeypatch):

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "select_prompt", lambda message, choices: "home")

    @click.command()
    @click.argument(
        "button", cls=menu.ChoiceArgument, type=click.Choice(["back", "home"])
    )
    def press(button):
        click.echo(button)

    result = CliRunner().invoke(press, [])
    assert result.exit_code == 0
    assert "home" in result.output
