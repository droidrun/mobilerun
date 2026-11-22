import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import click
import pytest
from click.testing import CliRunner
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from mobilerun.cli.base import prompts
from mobilerun.cli.base.prompts import SelectChoice, select_prompt_async
from mobilerun.cli.devices.app.uninstall import uninstall
from mobilerun.cli.devices.base import pickers
from mobilerun.cli.devices.base.options import DeviceTarget


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
            application = prompt._application if kind == "select" else prompt._session.app
            task = asyncio.create_task(prompts._execute_async(prompt))
            await asyncio.sleep(0.02)
            assert application.is_running
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=0.3)
            assert not application.is_running
            assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]

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
                assert await select_prompt_async("Select", [SelectChoice("one", "One")]) is None

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
        prompt = SimpleNamespace(execute=lambda: pytest.fail("sync execute must not run"))
        with pytest.raises(click.ClickException, match="Interactive prompt failed"):
            prompts._execute(prompt)

    asyncio.run(run())
