"""Completion installation stays safe, idempotent, and shell-aware."""

import importlib
from pathlib import Path

import pytest
from click.testing import CliRunner

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


def test_non_tty_or_print_only_writes_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setattr(completion, "_stdout_is_tty", lambda: False, raising=False)
    result = CliRunner().invoke(completion.completion, ["zsh"])
    assert result.exit_code == 0, result.output
    assert "_MOBILERUN_COMPLETE" in result.output
    assert not (tmp_path / ".mobilerun").exists()
    monkeypatch.setattr(completion, "_stdout_is_tty", lambda: True, raising=False)
    result = CliRunner().invoke(completion.completion, ["zsh", "--print"])
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
    from mobilerun.cli.app import cli
    from mobilerun.cli.base import menu

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
