import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from mobilerun.cli.devices.group import device as cli
from mobilerun.core.devices.base import Device
from mobilerun.core.extensions import Extension, action
from mobilerun.models.context import ContextItem


class DemoExtension(Extension):
    default_name = "demo"
    exclusive = "example"

    async def context(self):
        return [ContextItem("demo", "info", {"message": "ready"})]

    @action()
    async def repeat(self, text: str, count: int, loud: bool = False) -> str:
        """Repeat a message."""
        result = text * count
        return result.upper() if loud else result


@pytest.fixture
def connected(monkeypatch):
    from mobilerun.cli.devices.base import selection as extensions

    device = Device()
    asyncio.run(DemoExtension().bind(device))
    device.close = AsyncMock()
    monkeypatch.setattr(extensions, "connect_target", AsyncMock(return_value=device))
    return device


def test_extensions_json_reports_context(connected):
    result = CliRunner().invoke(
        cli, ["action", "extensions", "emulator-5554", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [
        {
            "name": "demo",
            "kind": "DemoExtension",
            "exclusive": "example",
            "context": [
                {"source": "demo", "kind": "info", "data": {"message": "ready"}}
            ],
        }
    ]
    connected.close.assert_awaited_once()


def test_actions_lists_public_actions(connected):
    result = CliRunner().invoke(cli, ["action", "list", "emulator-5554"])
    assert result.exit_code == 0, result.output
    assert "demo.repeat" in result.output and "Repeat a message" in result.output
    connected.close.assert_awaited_once()


def test_call_converts_types_and_serializes(connected):
    result = CliRunner().invoke(
        cli,
        [
            "action",
            "call",
            "demo.repeat",
            "emulator-5554",
            "--arg",
            "text=ab",
            "--arg",
            "count=2",
            "--arg",
            "loud=true",
            "--json",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(result.output) == "ABAB"
    connected.close.assert_awaited_once()


def test_call_missing_arg_is_clean_error(connected):
    result = CliRunner().invoke(
        cli, ["action", "call", "demo.repeat", "emulator-5554", "--arg", "text=x"]
    )
    assert result.exit_code == 1
    assert "count" in result.output
    connected.close.assert_awaited_once()


def test_call_prompts_required_argument(connected, monkeypatch):
    import importlib

    extensions = importlib.import_module("mobilerun.cli.devices.action.call")

    monkeypatch.setattr(extensions, "is_interactive", lambda: True)
    monkeypatch.setattr(extensions, "text_prompt", AsyncMock(return_value="2"))
    result = CliRunner().invoke(
        cli,
        ["action", "call", "demo.repeat", "emulator-5554", "--arg", "text=x", "--json"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(result.output) == "xx"


def test_call_rejects_unknown_action_and_closes(connected):
    result = CliRunner().invoke(
        cli, ["action", "call", "demo.missing", "emulator-5554"]
    )
    assert result.exit_code == 1
    assert result.output.strip() == "Error: Unknown action: demo.missing"
    connected.close.assert_awaited_once()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--arg", "oops"],
        ["--arg", "unknown=x"],
        ["--arg", "count=bad", "--arg", "text=x"],
    ],
)
def test_call_bad_arguments_are_clean_errors(connected, arguments):
    result = CliRunner().invoke(
        cli, ["action", "call", "demo.repeat", "emulator-5554", *arguments]
    )
    assert result.exit_code == 1
    assert result.output.startswith("Error: ")
    connected.close.assert_awaited_once()


def test_connection_error_is_one_line(monkeypatch):
    from mobilerun.cli.devices.base import selection as extensions

    monkeypatch.setattr(
        extensions,
        "connect_target",
        AsyncMock(side_effect=RuntimeError("offline\ntry later")),
    )
    result = CliRunner().invoke(cli, ["action", "list", "emulator-5554"])
    assert result.exit_code == 1
    assert "--debug" in result.output
