"""Confirmation prompts preserve safe defaults in synchronous and async commands."""

import asyncio
from unittest.mock import AsyncMock

import click
from click.testing import CliRunner

from mobilerun.cli.base import menu, prompts, safety


def test_sync_confirmation_default_and_escape(monkeypatch):
    seen = []

    def select(message, choices, *, default):
        seen.append((message, [choice.value for choice in choices], default))
        return None

    monkeypatch.setattr(prompts, "select_prompt", select)
    assert prompts.confirm_prompt("Continue?") is False
    assert seen == [("Continue?", ["yes", "no"], "no")]


def test_sync_confirmation_yes_with_default_yes(monkeypatch):
    monkeypatch.setattr(
        prompts, "select_prompt", lambda message, choices, *, default: default
    )
    assert prompts.confirm_prompt("Replace?", default=True) is True


def test_async_safety_uses_prompt_and_yes_skips(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    prompt = AsyncMock(return_value=True)
    monkeypatch.setattr(safety, "confirm_prompt_async", prompt, raising=False)

    @click.command()
    def command():
        async def invoke():
            await safety.confirm_action_async("delete", "paid", "Warning", yes=False)
            await safety.confirm_action_async("delete", "paid", "Warning", yes=True)

        asyncio.run(invoke())

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    prompt.assert_awaited_once_with("Continue?", default=False)


def test_async_safety_noninteractive_never_reads_prompt(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: False)
    prompt = AsyncMock()
    monkeypatch.setattr(safety, "confirm_prompt_async", prompt, raising=False)

    @click.command()
    def command():
        asyncio.run(safety.confirm_action_async("delete", "paid", "Warning", yes=False))

    result = CliRunner().invoke(command, input="y\n")
    assert result.exit_code == 1
    assert "--yes" in result.output
    prompt.assert_not_awaited()


def test_unmanaged_proxy_conflict_is_specific_extension_conflict():
    from mobilerun.core.proxy import UnmanagedProxyConflict
    from mobilerun.errors import ExtensionConflict
    from mobilerun.errors import UnmanagedProxyConflict as PublicConflict

    assert issubclass(UnmanagedProxyConflict, ExtensionConflict)
    assert UnmanagedProxyConflict is PublicConflict


def test_sync_safety_decline_exits_zero(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    observed = []
    monkeypatch.setattr(
        safety, "confirm_prompt", lambda message, **kw: observed.append(kw) or False
    )

    @click.command()
    def command():
        safety.confirm_action("delete", "paid", "Warning", yes=False)
        click.echo("should not run")

    result = CliRunner().invoke(command)
    assert result.exit_code == 0
    assert "should not run" not in result.output
    assert observed == [{"default": False}]


def test_async_safety_decline_exits_zero(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(safety, "confirm_prompt_async", AsyncMock(return_value=False))

    @click.command()
    def command():
        asyncio.run(safety.confirm_action_async("delete", "paid", "Warning", yes=False))
        click.echo("should not run")

    result = CliRunner().invoke(command)
    assert result.exit_code == 0
    assert "should not run" not in result.output


def test_json_redirect_does_not_make_stdout_noninteractive_look_interactive(
    monkeypatch,
):
    import sys

    from mobilerun.cli.proxy import group

    buy = AsyncMock()
    monkeypatch.setattr(group, "buy_proxy", buy)
    monkeypatch.setattr(safety, "confirm_prompt_async", AsyncMock(return_value=True))

    def check_ttys():
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
        if sys.stdout is not sys.stderr:
            monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        return sys.stdin.isatty() and sys.stdout.isatty()

    monkeypatch.setattr(menu, "is_interactive", check_ttys)
    result = CliRunner().invoke(group.proxy, ["buy", "--country", "US", "--json"])
    assert result.exit_code != 0
    assert "--yes" in result.output
    buy.assert_not_awaited()
    safety.confirm_prompt_async.assert_not_awaited()


def test_sync_safety_running_loop_explains_async_fix(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(prompts, "_select", lambda *args, **kwargs: object())

    @click.command()
    def command():
        async def run():
            safety.confirm_action("delete", "paid", "Warning", yes=False)

        asyncio.run(run())

    result = CliRunner().invoke(command)
    assert result.exit_code == 1
    assert "confirm_action_async()" in result.output
    assert "_async" in result.output


def test_sync_safety_inside_loop_skips_prompt_for_yes_and_noninteractive(monkeypatch):
    monkeypatch.setattr(
        prompts,
        "_select",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prompt created")),
    )

    @click.command()
    @click.option("--yes", is_flag=True)
    def command(yes):
        async def run():
            safety.confirm_action("delete", "paid", "Warning", yes=yes)

        asyncio.run(run())

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    accepted = CliRunner().invoke(command, ["--yes"])
    assert accepted.exit_code == 0, accepted.output
    monkeypatch.setattr(menu, "is_interactive", lambda: False)
    refused = CliRunner().invoke(command)
    assert refused.exit_code == 1
    assert "--yes" in refused.output
