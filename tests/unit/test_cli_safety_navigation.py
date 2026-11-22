"""Safety and navigation behavior at the CLI boundary."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.base import menu


def test_back_returns_to_parent_and_root_says_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    selections = iter(["child", "__back__", "__back__"])
    seen: list[list[str]] = []

    def select(message: str, choices: list[menu.SelectChoice]) -> str:
        seen.append([choice.label for choice in choices])
        return next(selections)

    monkeypatch.setattr(menu, "select_prompt", select)

    @click.group(cls=menu.MenuGroup)
    def root() -> None:
        pass

    @root.group()
    def child() -> None:
        pass

    @child.command()
    def leaf() -> None:
        pass

    result = CliRunner().invoke(root, [])
    assert result.exit_code == 0, result.output
    assert len(seen) == 3
    assert seen[0][-1] == "Exit" and seen[1][-1] == "Back"


def test_unknown_command_suggests_nearest_match() -> None:
    @click.group(cls=menu.MenuGroup)
    def group() -> None:
        pass

    @group.command()
    def device() -> None:
        pass

    result = CliRunner().invoke(group, ["devce"])
    assert result.exit_code == 2
    assert "Did you mean 'device'?" in result.output


def test_proxy_buy_requires_yes_in_non_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    from mobilerun.cli.proxy import group

    buy = AsyncMock(return_value=SimpleNamespace(id="p1"))
    monkeypatch.setattr(group, "buy_proxy", buy)
    monkeypatch.setattr(menu, "is_interactive", lambda: False)
    result = CliRunner().invoke(group.proxy, ["buy", "--country", "DE"])
    assert result.exit_code == 1
    assert "Refusing to buy proxy without confirmation; pass --yes." in result.output
    buy.assert_not_awaited()
    result = CliRunner().invoke(group.proxy, ["buy", "--country", "DE", "--yes"])
    assert result.exit_code == 0, result.output
    assert "✓ Bought proxy p1 (DE)" in result.output
    buy.assert_awaited_once()


def test_proxy_status_json_redacts_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    from mobilerun.cli.proxy import group

    monkeypatch.setattr(
        group,
        "get_proxy",
        AsyncMock(
            return_value=SimpleNamespace(
                id="p1", country="DE", status="active", type="mobile", password="secret"
            )
        ),
    )
    result = CliRunner().invoke(group.proxy, ["status", "p1", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "id": "p1",
        "country": "DE",
        "status": "active",
        "type": "mobile",
    }
    assert "secret" not in result.output


def test_completion_outputs_shell_script(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    from mobilerun.cli.app import cli

    monkeypatch.setattr(
        importlib.import_module("mobilerun.cli.base.completion"),
        "_bash_version",
        lambda: (5, 2),
    )
    result = CliRunner().invoke(cli, ["completion", "bash"])
    assert result.exit_code == 0, result.output
    assert "_MOBILERUN_COMPLETE" in result.output
    assert "_mobilerun_completion" in result.output


@pytest.mark.parametrize(
    "module_name,command,args,action",
    [
        ("mobilerun.cli.mailbox.group", "mailbox", ["create"], "create mailbox"),
        ("mobilerun.cli.phone_number.group", "number", ["create"], "create number"),
        (
            "mobilerun.cli.mailbox.group",
            "mailbox",
            ["release", "id1"],
            "release mailbox",
        ),
        (
            "mobilerun.cli.phone_number.group",
            "number",
            ["release", "id1"],
            "release number",
        ),
    ],
)
def test_resource_actions_refuse_noninteractive_without_yes(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    command: str,
    args: list[str],
    action: str,
) -> None:
    from importlib import import_module

    monkeypatch.setattr(menu, "is_interactive", lambda: False)
    group = getattr(import_module(module_name), command)
    result = CliRunner().invoke(group, args)
    assert result.exit_code == 1, result.output
    assert f"Refusing to {action} without confirmation; pass --yes." in result.output


def test_interactive_buy_defaults_no_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    from mobilerun.cli.proxy import group

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    from mobilerun.cli.base import safety

    prompt = AsyncMock(return_value=False)
    monkeypatch.setattr(safety, "confirm_prompt_async", prompt)
    buy = AsyncMock(return_value=SimpleNamespace(id="p1"))
    monkeypatch.setattr(group, "buy_proxy", buy)
    result = CliRunner().invoke(group.proxy, ["buy", "--country", "DE"], input="n\n")
    assert result.exit_code == 0, result.output
    assert "Billing: paid. This may incur charges." in result.output
    prompt.assert_awaited_once_with("Continue?", default=False)
    buy.assert_not_awaited()


def test_auth_status_json_masks_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from mobilerun.cli.auth import status

    monkeypatch.setattr(status.credentials, "load_credentials", lambda: None)
    monkeypatch.setenv("MOBILERUN_API_KEY", "abcdefghijk")
    result = CliRunner().invoke(status.whoami, ["--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"source": "environment", "api_key": "abc…hijk"}
    assert "abcdefghijk" not in result.output


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_completion_works_for_each_shell(
    shell: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    from mobilerun.cli.app import cli

    monkeypatch.setattr(
        importlib.import_module("mobilerun.cli.base.completion"),
        "_bash_version",
        lambda: (5, 2),
    )
    result = CliRunner().invoke(cli, ["completion", shell])
    assert result.exit_code == 0, result.output
    assert "_MOBILERUN_COMPLETE" in result.output
    assert "_mobilerun_completion" in result.output


def test_escape_returns_to_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    replies = iter(["child", None, "__back__"])
    monkeypatch.setattr(menu, "select_prompt", lambda message, choices: next(replies))

    @click.group(cls=menu.MenuGroup)
    def group() -> None:
        pass

    @group.group()
    def child() -> None:
        pass

    @child.command()
    def leaf() -> None:
        pass

    assert CliRunner().invoke(group, []).exit_code == 0
    with pytest.raises(StopIteration):
        next(replies)


def test_ctrl_c_from_menu_exits_130(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    def interrupt(message: str, choices: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(menu, "select_prompt", interrupt)

    @click.group(cls=menu.MenuGroup)
    def group() -> None:
        pass

    @group.command()
    def leaf() -> None:
        pass

    assert CliRunner().invoke(group, []).exit_code == 130


def test_root_help_contains_examples() -> None:
    from mobilerun.cli.app import cli

    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    from mobilerun.addons.agents import agents_available

    for example in ("mobilerun device list", "mobilerun skills install"):
        assert example in result.output
    if agents_available():
        assert "mobilerun agent run" in result.output
    else:
        assert "mobilerun agent run" not in result.output


def test_release_warns_and_yes_skips_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mobilerun.cli.mailbox import group

    release = AsyncMock(
        return_value=SimpleNamespace(data=SimpleNamespace(status="cancel_scheduled"))
    )
    monkeypatch.setattr(
        group, "Mailbox", lambda *args, **kwargs: SimpleNamespace(release=release)
    )
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    from mobilerun.cli.base import safety

    prompt = AsyncMock(return_value=False)
    monkeypatch.setattr(safety, "confirm_prompt_async", prompt)
    denied = CliRunner().invoke(group.mailbox, ["release", "id1"], input="\n")
    assert denied.exit_code == 0, denied.output
    assert "This will release id1." in denied.output
    release.assert_not_awaited()
    accepted = CliRunner().invoke(group.mailbox, ["release", "id1", "--yes"])
    assert accepted.exit_code == 0, accepted.output
    assert "✓ Mailbox id1: cancel_scheduled" in accepted.output
    release.assert_awaited_once()
