"""Interactive prompts, menu navigation, and safe defaults in automation."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import click
import pytest
from click.testing import CliRunner
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from mobilerun.cli.base import menu, prompts, safety
from mobilerun.cli.base.prompts import SelectChoice, select_prompt_async
from mobilerun.cli.credentials import group as credentials
from mobilerun.cli.devices.app.uninstall import uninstall
from mobilerun.cli.devices.base import pickers
from mobilerun.cli.devices.base.options import DeviceTarget
from mobilerun.cli.proxy import group as proxy


def test_select_inside_running_event_loop():
    async def run():
        with (
            create_pipe_input() as input,
            create_app_session(input=input, output=DummyOutput()),
        ):
            input.send_text("\r")
            return await select_prompt_async("Select", [SelectChoice("one", "One")])

    assert asyncio.run(run()) == "one"


def test_uninstall_interactive_app_picker(monkeypatch):
    device = SimpleNamespace(
        list_apps=AsyncMock(
            return_value=[{"package": "com.example", "label": "Example"}]
        ),
        uninstall_app=AsyncMock(return_value="Success"),
    )

    @asynccontextmanager
    async def open_device(self):
        yield device

    monkeypatch.setattr(DeviceTarget, "open", open_device)
    monkeypatch.setattr(pickers, "is_interactive", lambda: True)
    with (
        create_pipe_input() as input,
        create_app_session(input=input, output=DummyOutput()),
    ):
        input.send_text("\r")
        result = CliRunner().invoke(uninstall, ["--yes", "--device", "fake"])
    assert result.exit_code == 0, result.output
    device.uninstall_app.assert_awaited_once_with("com.example")


def test_prompt_failure_has_clear_error(monkeypatch):
    async def fail():
        raise RuntimeError("terminal unavailable")

    prompt = SimpleNamespace(execute_async=fail)
    with pytest.raises(click.ClickException, match="Interactive prompt failed") as exc:
        asyncio.run(async_call(prompt))
    assert str(exc.value.__cause__) == "terminal unavailable"


async def async_call(prompt):
    return await prompts._execute_async(prompt)


@pytest.mark.parametrize("kind", ["select", "text", "secret"])
def test_timed_cancellation_closes_real_prompt_promptly(kind):
    async def run():
        with (
            create_pipe_input() as input,
            create_app_session(input=input, output=DummyOutput()),
        ):
            prompt = (
                prompts._select("Select", [SelectChoice("one", "One")])
                if kind == "select"
                else prompts._text("Text", secret=kind == "secret")
            )
            application = (
                prompt._application if kind == "select" else prompt._session.app
            )
            task = asyncio.create_task(prompts._execute_async(prompt))
            await asyncio.sleep(0.02)
            assert application.is_running
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=0.3)
            assert not application.is_running
            assert not [
                t
                for t in asyncio.all_tasks()
                if t is not asyncio.current_task() and not t.done()
            ]

    asyncio.run(run())


@pytest.mark.parametrize("key, expected", [("\x1b", None), ("\x03", KeyboardInterrupt)])
def test_async_prompt_existing_cancel_keybindings(key, expected):
    async def run():
        with (
            create_pipe_input() as input,
            create_app_session(input=input, output=DummyOutput()),
        ):
            input.send_text(key)
            if expected is KeyboardInterrupt:
                with pytest.raises(KeyboardInterrupt):
                    await select_prompt_async("Select", [SelectChoice("one", "One")])
            else:
                assert (
                    await select_prompt_async("Select", [SelectChoice("one", "One")])
                    is None
                )

    asyncio.run(run())


@pytest.mark.parametrize("secret", [False, True])
def test_async_text_escape_retains_empty_cancel_result(secret):
    async def run():
        with (
            create_pipe_input() as input,
            create_app_session(input=input, output=DummyOutput()),
        ):
            input.send_text("\x1b")
            assert await prompts.text_prompt_async("Text", secret=secret) == ""

    asyncio.run(run())


def test_sync_prompt_rejects_running_loop_before_creating_coroutine():
    async def run():
        prompt = SimpleNamespace(
            execute=lambda: pytest.fail("sync execute must not run")
        )
        with pytest.raises(click.ClickException, match="Interactive prompt failed"):
            prompts._execute(prompt)

    asyncio.run(run())


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
    monkeypatch.setattr(
        connect_module, "text_prompt", AsyncMock(return_value="127.0.0.1:5555")
    )
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
                if path == "mobilerun" and name == "skills":
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

    from mobilerun.cli.devices.base import options
    from mobilerun.cli.devices.base import selection as tap_module
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
    result = CliRunner().invoke(device, ["input", "tap"])
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


def test_sandbox_deploy_requires_file_in_automation() -> None:
    from mobilerun.cli.app import cli

    result = CliRunner().invoke(cli, ["sandbox", "deploy"])
    assert result.exit_code == 2
    assert "Missing argument 'FILE'" in result.output


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
    from mobilerun.clients.cloud import proxies

    client = AsyncMock(side_effect=AssertionError("must not construct a client"))
    monkeypatch.setattr(proxies, "cloud_client", client)
    monkeypatch.setattr(safety, "confirm_prompt_async", AsyncMock(return_value=True))

    def check_ttys():
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
        if sys.stdout is not sys.stderr:
            monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        return sys.stdin.isatty() and sys.stdout.isatty()

    monkeypatch.setattr(menu, "is_interactive", check_ttys)
    result = CliRunner().invoke(
        group.proxy, ["buy", "--country", "US", "--json", "--billing", "included"]
    )
    assert result.exit_code != 0
    assert "--yes" in result.output
    client.assert_not_called()
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


@pytest.mark.parametrize("tty", [True, False])
def test_proxy_replacement_requires_opt_in(monkeypatch, tty):
    phone = SimpleNamespace(close=AsyncMock())
    extension = SimpleNamespace(
        bind=AsyncMock(
            side_effect=[proxy.UnmanagedProxyConflict("use force=True"), None]
        ),
        status=AsyncMock(return_value=SimpleNamespace(connected=True)),
    )
    prompt = AsyncMock(side_effect=lambda message, *, default: default)
    monkeypatch.setattr(proxy.menu, "is_interactive", lambda: tty)
    monkeypatch.setattr(proxy, "CloudDevice", lambda **kwargs: phone)
    monkeypatch.setattr(proxy, "CloudFixedProxy", lambda id: extension)
    monkeypatch.setattr(proxy, "confirm_prompt_async", prompt)
    result = CliRunner().invoke(proxy.proxy, ["bind", "new", "phone"])
    assert result.exit_code == (0 if tty else 1), result.output
    assert ("Cancelled." if tty else "--force") in result.output
    assert prompt.await_count == int(tty)
    extension.bind.assert_awaited_once_with(phone, force=False)
    phone.close.assert_awaited_once()


@pytest.mark.parametrize("tty", [True, False])
def test_confirmed_credential_delete_skips_field_selection(monkeypatch, tty):
    store = SimpleNamespace(
        list=AsyncMock(return_value=[]), delete=AsyncMock(), close=AsyncMock()
    )
    monkeypatch.setattr(credentials, "CloudCredentials", lambda **kwargs: store)
    monkeypatch.setattr(credentials, "is_interactive", lambda: tty)
    prompt = Mock(return_value=None)
    monkeypatch.setattr(credentials, "select_prompt", prompt)
    result = CliRunner().invoke(
        credentials.credentials, ["delete", "app", "work", "--yes"]
    )
    assert result.exit_code == 0, result.output
    prompt.assert_not_called()
    store.delete.assert_awaited_once_with("app", "work", None)


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


@pytest.mark.parametrize(
    "module_name,command,args,action",
    [
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
    from mobilerun.addons.plugins import discover
    from mobilerun.addons.settings import CLI_ENTRY_POINT_GROUP

    for example in ("mobilerun device list", "mobilerun skills install"):
        assert example in result.output
    if "agent" in discover(CLI_ENTRY_POINT_GROUP):
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
    assert "Release 1 mailbox: id1." in denied.output
    release.assert_not_awaited()
    accepted = CliRunner().invoke(group.mailbox, ["release", "id1", "--yes"])
    assert accepted.exit_code == 0, accepted.output
    assert "✓ Mailbox id1: cancel_scheduled" in accepted.output
    release.assert_awaited_once()
