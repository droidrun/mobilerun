"""Destructive compute commands share prompting and automation policy."""

import json
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from test_cli_function_cloud import ID2, sdk  # noqa: F401

from mobilerun.cli import options
from mobilerun.cli.main import cli

COMMANDS = [
    ["function", "delete", ID2, "--all"],
    ["routine", "delete", ID2],
    ["sandbox", "delete", ID2],
    ["sandbox", "stop", ID2],
]


@pytest.mark.parametrize("command", COMMANDS)
def test_interactive_confirmation_uses_one_prompt(
    sdk, monkeypatch, command  # noqa: F811
):
    monkeypatch.setattr(options, "input_allowed", lambda no_input: not no_input)
    result = CliRunner().invoke(cli, [*command, "-j"], input="y\n")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["id"] == ID2
    assert result.stderr.count("[y/N]") == 1
    sdk.factory.assert_called_once()


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize(
    "flags,input_text,expected,message",
    [
        ([], "n\n", 1, "cancelled"),
        (["--no-input"], "y\n", 2, "--yes"),
    ],
)
def test_cancel_or_no_input_prevents_remote_delete(
    sdk, monkeypatch, command, flags, input_text, expected, message  # noqa: F811
):
    monkeypatch.setattr(options, "input_allowed", lambda no_input: not no_input)
    result = CliRunner().invoke(cli, [*command, *flags], input=input_text)
    assert result.exit_code == expected, result.output
    assert message in result.stderr
    # `function delete` resolves the name before it asks; nothing is deleted.
    sdk.functions.delete.assert_not_called()
    sdk.functions.versions.delete.assert_not_called()
    sdk.client.compute.sandboxes.delete.assert_not_called()
    sdk.client.workflows.flows.delete.assert_not_called()


@pytest.mark.parametrize("command", COMMANDS)
def test_yes_overrides_no_input_without_prompt(sdk, monkeypatch, command):  # noqa: F811
    monkeypatch.setattr(
        options.click, "confirm", Mock(side_effect=AssertionError("must not prompt"))
    )
    result = CliRunner().invoke(cli, [*command, "--yes", "--no-input", "-j"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["id"] == ID2
    sdk.factory.assert_called_once()
