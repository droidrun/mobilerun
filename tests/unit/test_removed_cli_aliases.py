"""Removed aliases are unknown while the current command tree remains usable."""

import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli


@pytest.mark.parametrize(
    "name",
    [
        "run",
        "configure",
        "devices",
        "login",
        "logout",
        "whoami",
        "connect",
        "disconnect",
        "setup",
        "ping",
        "auth",
    ],
)
def test_removed_root_alias(name):
    result = CliRunner().invoke(cli, [name, "--help"])
    assert result.exit_code == 2
    assert f"No such command '{name}'" in result.output


def test_unknown_alias_suggests_current_command():
    result = CliRunner().invoke(cli, ["devices"])
    assert "Did you mean 'device'?" in result.output


def test_removed_number_sms_alias():
    result = CliRunner().invoke(cli, ["number", "sms", "--help"])
    assert result.exit_code == 2
    assert "No such command 'sms'" in result.output


@pytest.mark.parametrize(
    "name",
    [
        "screenshot",
        "ui",
        "tap",
        "swipe",
        "long-press",
        "type",
        "press",
        "apps",
        "start",
    ],
)
def test_removed_device_alias(name):
    result = CliRunner().invoke(cli, ["device", name, "--help"])
    assert result.exit_code == 2
    assert f"No such command '{name}'" in result.output


@pytest.mark.parametrize(
    "path",
    [
        ["account", "login"],
        ["account", "logout"],
        ["account", "status"],
        ["device", "list"],
        ["device", "connect"],
        ["device", "disconnect"],
        ["device", "setup"],
        ["device", "ping"],
        ["device", "input", "tap"],
    ],
)
def test_current_command_help(path):
    result = CliRunner().invoke(cli, [*path, "--help"])
    assert result.exit_code == 0, result.output
