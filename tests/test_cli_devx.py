"""User-facing help, output, and automation contracts for the CLI."""

import importlib
from unittest.mock import AsyncMock, Mock

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.main import cli
from mobilerun.config_manager import ConfigLoader

run_module = importlib.import_module("mobilerun.cli.run")


def commands(command=cli, path=()):
    yield path, command
    if isinstance(command, click.Group):
        for name, child in command.commands.items():
            yield from commands(child, (*path, name))


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr(ConfigLoader, "get_user_config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setenv("MOBILERUN_TELEMETRY_ENABLED", "false")
    for name in ("MOBILERUN_CONFIG", "MOBILERUN_TARGET", "MOBILERUN_DEVICE", "MOBILERUN_DEBUG"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("path", [(), ("run",), ("device",), ("device", "list"),
                                 ("device", "ui"), ("device", "pair"), ("auth", "login"),
                                 ("configure",), ("macro", "replay")])
def test_help_has_copyable_examples(path):
    result = CliRunner().invoke(cli, [*path, "--help"])
    assert result.exit_code == 0, result.output
    assert "Examples:\n    mobilerun " in result.stdout
    assert "\n    mobilerun " in result.stdout
    if not path:
        assert result.stdout.rstrip().endswith("Docs: https://docs.mobilerun.ai/framework/guides/cli")


@pytest.mark.parametrize("path,command", list(commands()))
def test_help_options_and_summaries_are_complete(path, command):
    with click.Context(command, **command.context_settings) as ctx:
        params = command.get_params(ctx)
        for param in params:
            if isinstance(param, click.Option):
                assert param.help and param.help.endswith("."), (path, param.name)
                assert param.show_default is not False, (path, param.name)
        names = [param.name for param in params]
        shared_first = [name for name in ("device", "target", "platform", "config_path") if name in names]
        shared_last = [name for name in ("api_url", "json_output", "verbose", "help") if name in names]
        assert names[:len(shared_first)] == shared_first
        assert names[-len(shared_last):] == shared_last
    assert command.get_short_help_str(limit=1).endswith("."), path
    assert not command.get_short_help_str(limit=1).endswith("..."), path


def test_root_command_order_and_agent_discovery(monkeypatch):
    monkeypatch.setattr(run_module, "list_agents", Mock(side_effect=AssertionError("root must not scan")))
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    names = ["run", "device", "configure", "auth", "doctor", "env", "macro"]
    assert [result.stdout.index(f"  {name} ") for name in names] == sorted(
        result.stdout.index(f"  {name} ") for name in names
    )


def test_agent_option_tracks_installed_agents(monkeypatch):
    monkeypatch.setattr(run_module, "list_agents", lambda: [])
    runner = CliRunner()
    assert "--agent" not in runner.invoke(cli, ["run", "--help"]).stdout
    result = runner.invoke(cli, ["run", "TASK", "-a", "missing"])
    assert result.exit_code == 2
    assert "No external agents are installed; install one before using --agent." in result.stderr
    monkeypatch.setattr(run_module, "list_agents", lambda: ["installed"])
    assert "--agent [installed]" in runner.invoke(cli, ["run", "--help"]).stdout
    callback = AsyncMock(return_value=(True, "done", 1))
    monkeypatch.setattr(run_module, "_run_task", callback)
    assert runner.invoke(cli, ["run", "TASK", "-a", "installed"]).exit_code == 0
    assert callback.await_args.kwargs["agent"] == "installed"
    assert runner.invoke(cli, ["run", "TASK", "-a", "missing"]).exit_code == 2


@pytest.mark.parametrize("args,suggestion", [(["devcie"], "device"), (["device", "lst"], "list"),
                                             (["auth", "stauts"], "status")])
def test_command_typos_suggest_without_accepting(args, suggestion):
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 2
    assert f"Did you mean '{suggestion}'?" in result.stderr


def test_no_suggestion_for_unrelated_command():
    result = CliRunner().invoke(cli, ["unrelated-nonsense"])
    assert result.exit_code == 2
    assert "Did you mean" not in result.stderr


@pytest.mark.parametrize("path", [("run",), ("configure",), ("macro", "replay")])
def test_provider_help_wraps_and_configure_only_lists_supported(path):
    result = CliRunner().invoke(cli, [*path, "--help"])
    assert "-p, --provider PROVIDER" in result.stdout
    assert max(map(len, result.stdout.splitlines())) < 100
    if path == ("configure",):
        assert "openrouter" not in result.stdout and "deepseek" not in result.stdout
        for name in ("openrouter", "deepseek"):
            assert CliRunner().invoke(cli, ["configure", "-p", name]).exit_code == 2


def test_login_and_connection_usage():
    runner = CliRunner()
    assert "[SERVICE]" in runner.invoke(cli, ["auth", "login", "--help"]).stdout
    for command in ("connect", "disconnect"):
        assert "HOST:PORT" in runner.invoke(cli, ["device", command, "--help"]).stdout
    text = runner.invoke(cli, ["auth", "login", "--help"]).stdout
    assert "[default: 300.0]" in text
    assert "(openai, anthropic, gemini)" in " ".join(text.split())
