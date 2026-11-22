"""Regression coverage for required command parameters in interactive shells."""

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.base import menu


def test_direct_command_prompts_for_missing_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "text_prompt", lambda message: "42", raising=False)

    @click.group(cls=menu.MenuGroup)
    def group() -> None:
        pass

    @group.command()
    @click.argument("x", type=int)
    def tap(x: int) -> None:
        click.echo(f"{x}:{type(x).__name__}")

    result = CliRunner().invoke(group, ["tap"])
    assert result.exit_code == 0, result.output
    assert "42:int" in result.output


def test_required_option_uses_typed_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(
        menu, "select_prompt", lambda message, choices: choices[0].value
    )

    @click.group(cls=menu.MenuGroup)
    def group() -> None:
        pass

    @group.command()
    @click.option(
        "--device",
        required=True,
        cls=menu.PromptingOption,
        choices=lambda ctx: [menu.SelectChoice("serial-1", "Device")],
    )
    def show(device: str) -> None:
        click.echo(device)

    result = CliRunner().invoke(group, ["show"])
    assert result.exit_code == 0, result.output
    assert "serial-1" in result.output


@pytest.mark.parametrize("interrupted,expected", [(False, 0), (True, 130)])
def test_prompt_cancel(
    monkeypatch: pytest.MonkeyPatch, interrupted: bool, expected: int
) -> None:
    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    def select(message: str, choices: object) -> None:
        if interrupted:
            raise KeyboardInterrupt
        return None

    monkeypatch.setattr(menu, "select_prompt", select)

    @click.group(cls=menu.MenuGroup)
    def group() -> None:
        pass

    @group.command()
    @click.argument("value", type=click.Choice(["one"]))
    def show(value: str) -> None:
        raise AssertionError("cancelled command ran")

    assert CliRunner().invoke(group, ["show"]).exit_code == expected


def test_non_tty_retains_missing_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(menu, "is_interactive", lambda: False)

    @click.group(cls=menu.MenuGroup)
    def group() -> None:
        pass

    @group.command()
    @click.argument("x", type=int)
    def tap(x: int) -> None:
        pass

    result = CliRunner().invoke(group, ["tap"])
    assert result.exit_code == 2
    assert "Missing argument 'X'" in result.output


def test_menu_connect_prompts_address(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices import connect as connect_module
    from mobilerun.cli.devices.group import device

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "select_prompt", lambda message, choices: "connect")
    monkeypatch.setattr(connect_module, "is_interactive", lambda: True)
    monkeypatch.setattr(connect_module, "discover_devices", AsyncMock(return_value=[]))
    seen = []

    def select(message, choices):
        seen.extend(choice.label for choice in choices)
        return "__add_address__"

    monkeypatch.setattr(connect_module, "select_prompt", AsyncMock(side_effect=select))
    monkeypatch.setattr(connect_module, "text_prompt", AsyncMock(return_value="127.0.0.1:5555"))
    connect = AsyncMock(return_value="connected to 127.0.0.1:5555")
    monkeypatch.setattr(connect_module.adb, "connect", connect)
    saved = []
    monkeypatch.setattr(connect_module, "save_default_device", saved.append)
    result = CliRunner().invoke(device, [])
    assert result.exit_code == 0, result.output
    assert seen[-1] == "Add device by address (host:port)…"
    assert saved == ["127.0.0.1:5555"]
    connect.assert_awaited_once_with("127.0.0.1:5555")


def test_cli_command_tree_required_parameters_are_prompting() -> None:
    from mobilerun.cli.app import cli

    def walk(command: click.Command, path: str) -> None:
        menu.prepare_command(command)
        for param in command.params:
            if param.required:
                assert isinstance(
                    param, menu.PromptingParameter
                ), f"{path}: {param.name}"
        if isinstance(command, click.Group):
            assert isinstance(command, menu.MenuGroup), path
            ctx = click.Context(command, info_name=command.name)
            for name in command.list_commands(ctx):
                if path == "mobilerun" and name in {"skills", "sandbox"}:
                    continue
                child = command.get_command(ctx, name)
                assert child is not None
                walk(child, f"{path} {name}")

    walk(cli, "mobilerun")


def test_disconnect_clears_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices import disconnect as disconnect_module
    from mobilerun.cli.devices.group import device

    monkeypatch.setattr(
        disconnect_module.ConfigLoader,
        "load",
        lambda _: SimpleNamespace(device=SimpleNamespace(serial="serial-1")),
    )
    disconnect = AsyncMock()
    monkeypatch.setattr(disconnect_module.adb, "disconnect", disconnect)
    saved = []
    monkeypatch.setattr(disconnect_module, "save_default_device", saved.append)
    result = CliRunner().invoke(device, ["disconnect"])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "✓ Disconnected serial-1"
    assert saved == [None]
    disconnect.assert_not_awaited()


def test_cloud_resource_choices_use_no_commerce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mobilerun.cli.base import choices
    from mobilerun.models.billing import Billing

    list_mailboxes = AsyncMock(
        return_value=[SimpleNamespace(id="id-1", address="mail@example.test")]
    )
    monkeypatch.setattr(choices, "list_mailboxes", list_mailboxes)
    result = choices.mailboxes(click.Context(click.Command("test")))
    assert result == [menu.SelectChoice("id-1", "mail@example.test")]
    list_mailboxes.assert_awaited_once_with(billing=Billing.NO_COMMERCE)


def test_direct_tap_prompts_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices.base import selection as tap_module
    from mobilerun.cli.devices.base import options
    from mobilerun.cli.devices.group import device

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    labels: list[str] = []

    def text(message: str) -> str:
        labels.append(message)
        return {"X coordinate": "12", "Y coordinate": "34"}[message]

    monkeypatch.setattr(menu, "text_prompt", text)
    phone = AsyncMock()
    monkeypatch.setattr(tap_module, "connect_target", AsyncMock(return_value=phone))
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(
            return_value=[
                SimpleNamespace(
                    id="serial-1", name=None, source="local", state=None, ready=True
                )
            ]
        ),
    )
    result = CliRunner().invoke(device, ["tap"])
    assert result.exit_code == 0, result.output
    assert labels == ["X coordinate", "Y coordinate"]
    phone.tap.assert_awaited_once_with(12, 34)


def test_required_text_cancels_after_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    replies = iter(["  ", "  host:5555  "])
    monkeypatch.setattr(menu, "text_prompt", lambda message: next(replies))

    @click.group(cls=menu.MenuGroup)
    def group() -> None:
        pass

    @group.command()
    @click.argument("address")
    def connect(address: str) -> None:
        click.echo(f"connected:{address}")

    result = CliRunner().invoke(group, ["connect"])
    assert result.exit_code == 0, result.output
    assert "Cancelled." in result.output
    assert "connected:" not in result.output


def test_sandbox_is_not_a_cli_command() -> None:
    from mobilerun.cli.app import cli

    result = CliRunner().invoke(cli, ["sandbox"])
    assert result.exit_code == 2
    assert "No such command" in result.output
