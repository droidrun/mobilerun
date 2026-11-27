"""Macro CLI replay, discovery, logging, and privacy contracts."""

import asyncio
import importlib
import json
import logging
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.cli.devices.base import discovery, options, selection
from mobilerun.cli.devices.base.options import device_options
from mobilerun.macro.cli import macro
from mobilerun.macro.utils import configure_macro_logging
from mobilerun.models.results import ExecutionResult
from tests.support.macro import FakeDevice, write_macro


def test_macro_group_exposes_list_and_replay():

    result = CliRunner().invoke(macro, ["--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "replay" in result.output


def test_macro_replay_discovery_and_exclusive_flags(tmp_path, monkeypatch):

    monkeypatch.setattr(options, "discover_devices", AsyncMock(return_value=[]))
    path = write_macro(tmp_path)
    runner = CliRunner()
    missing = runner.invoke(macro, ["replay", str(path)])
    assert missing.exit_code == 1
    assert "No devices found." in missing.output
    multiple = runner.invoke(
        macro, ["replay", str(path), "target", "--android", "--ios"]
    )
    assert multiple.exit_code == 2
    assert "Choose at most one device type flag" in multiple.output


def fake_device_factory(monkeypatch):
    device = FakeDevice()
    targets = []

    async def create(target, **kwargs):
        targets.append(target)
        return device

    async def close():
        pass

    device.close = close

    monkeypatch.setattr(selection, "connect_target", create)
    return device, targets


def test_macro_cli_replays_on_selected_device(monkeypatch, tmp_path):

    device, targets = fake_device_factory(monkeypatch)
    result = CliRunner().invoke(
        cli, ["macro", "replay", str(write_macro(tmp_path)), "serial", "--delay", "0"]
    )
    assert result.exit_code == 0, result.output
    assert device.taps == [(10, 20)]
    assert targets == ["serial"]


def test_macro_cli_mismatch_exits_nonzero_without_action(monkeypatch, tmp_path):

    device, _ = fake_device_factory(monkeypatch)
    result = CliRunner().invoke(
        cli,
        [
            "macro",
            "replay",
            str(write_macro(tmp_path, True)),
            "serial",
            "--state-timeout",
            "0",
        ],
    )
    assert result.exit_code == 1
    assert "mismatch at step 1 (before)" in result.output
    assert device.taps == []


def test_dry_run_does_not_connect(monkeypatch, tmp_path):

    device, targets = fake_device_factory(monkeypatch)
    result = CliRunner().invoke(
        cli, ["macro", "replay", str(write_macro(tmp_path)), "serial", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "tap" in result.output
    assert targets == []
    assert device.taps == []


def test_versioned_targets_list_and_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(tmp_path))
    folder = tmp_path / "routine"
    folder.mkdir()
    pin = write_macro(folder)
    current = folder / "v10.jsonl"
    current.write_text(
        '{"i":3,"action":"tap","args":{"x":11,"y":20}}\n'
        '{"i":8,"action":"wait","args":{"duration":0}}\n'
    )
    (folder / "v2.jsonl").write_text(pin.read_text())
    (tmp_path / "other").mkdir()
    runner = CliRunner()
    listing = runner.invoke(macro, ["list", "--json"])
    assert listing.exit_code == 0, listing.output
    assert json.loads(listing.output) == [
        {"name": "routine", "version": "v10.jsonl", "actions": 2}
    ]
    for target in ("routine", str(folder), str(current)):
        result = runner.invoke(
            macro, ["replay", target, "--dry-run", "--start-from", "2"]
        )
        assert result.exit_code == 0, result.output
        assert result.output == "8: wait duration=0\n"
    result = runner.invoke(macro, ["replay", str(pin), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert result.output == "1: tap x=10 y=20\n"


@pytest.mark.parametrize("flags", [[], ["--yes"], ["--on-mismatch", "agent"]])
def test_invalid_macro_fails_before_device_discovery(tmp_path, monkeypatch, flags):
    discovery = AsyncMock(side_effect=AssertionError("must not discover"))
    monkeypatch.setattr(options, "discover_devices", discovery)
    path = tmp_path / "v1.jsonl"
    path.write_text("not a macro\n")
    result = CliRunner().invoke(macro, ["replay", str(path), *flags])
    assert result.exit_code == 1
    assert "Invalid JSONL row 1" in result.output
    discovery.assert_not_awaited()


def test_optional_device_with_no_discovery(monkeypatch):
    monkeypatch.setattr(
        "mobilerun.cli.devices.base.options.discover_devices",
        AsyncMock(return_value=[]),
    )

    @click.command()
    @device_options(optional=True)
    def command(**kwargs):
        click.echo(str(kwargs["target"]))

    result = CliRunner().invoke(command, [])
    assert result.exit_code == 0, result.output
    assert "None" in result.output


def test_adb_failure_does_not_hide_cloud_devices(monkeypatch):

    monkeypatch.setattr(
        discovery.adb, "list", AsyncMock(side_effect=RuntimeError("adb unavailable"))
    )
    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: "fake")
    monkeypatch.setattr(
        discovery,
        "list_devices",
        AsyncMock(
            return_value=[
                SimpleNamespace(
                    id="cloud-id", name="phone", state="ready", type="android"
                )
            ]
        ),
    )
    devices = asyncio.run(discovery.discover_devices())
    assert [d.id for d in devices] == ["cloud-id"]


def test_macro_dry_run_never_discovers_device(monkeypatch, tmp_path):

    discovery = AsyncMock(side_effect=AssertionError("must not discover"))
    monkeypatch.setattr(
        "mobilerun.cli.devices.base.options.discover_devices", discovery
    )
    path = write_macro(
        tmp_path, rows=[{"i": 1, "action": "wait", "args": {"duration": 0}}]
    )
    result = CliRunner().invoke(macro, ["replay", str(path), "--dry-run"])
    assert result.exit_code == 0, repr(result.exception)
    assert "wait" in result.output
    discovery.assert_not_called()


def test_replay_flags_are_accepted_in_dry_run(tmp_path):

    path = write_macro(tmp_path)
    result = CliRunner().invoke(
        macro,
        [
            "replay",
            str(path),
            "-d",
            "serial",
            "--on-mismatch",
            "agent",
            "--provider",
            "OpenAI",
            "--model",
            "foo",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output


def test_macro_module_entrypoint():

    result = subprocess.run(
        [sys.executable, "-m", "mobilerun.macro", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "replay" in result.stdout


def test_replay_config_reaches_device_connection(monkeypatch, tmp_path):

    replay_module = importlib.import_module("mobilerun.macro.cli.replay")

    config = SimpleNamespace(
        credentials={},
        device=SimpleNamespace(serial=None, platform="android", use_tcp=False),
    )
    monkeypatch.setattr(replay_module.ConfigLoader, "load", lambda *args: config)
    connect = AsyncMock(return_value=SimpleNamespace(close=AsyncMock()))
    monkeypatch.setattr(selection, "connect_target", connect)
    monkeypatch.setattr(
        replay_module.MacroOperator,
        "execute",
        AsyncMock(return_value=ExecutionResult(success=True, device="fake")),
    )
    path = write_macro(tmp_path)
    result = CliRunner().invoke(macro, ["replay", str(path), "-d", "serial"])
    assert result.exit_code == 0, repr(result.exception)
    assert connect.await_args.kwargs["config"] is config


def test_macro_agent_handoff_accepts_injected_runner(tmp_path):

    path = write_macro(tmp_path)
    received = []
    result = CliRunner().invoke(
        macro,
        ["replay", str(path), "--device", "serial", "--on-mismatch", "agent"],
        # Click children share metadata with their parent context.
        parent=click.Context(macro),
    )
    assert result.exit_code == 2
    assert "mobilerun macro replay" in result.output

    parent = click.Context(macro)
    parent.meta["macro_agent_handoff"] = lambda **kwargs: received.append(kwargs)
    result = CliRunner().invoke(
        macro,
        ["replay", str(path), "--device", "serial", "--on-mismatch", "agent"],
        parent=parent,
    )
    assert result.exit_code == 0, result.output
    assert received[0]["target"].identifier == "serial"
    assert received[0]["path"] == str(path)


def test_preview_shows_coordinates_without_snapshot_or_payload(tmp_path):
    secret = "PRIVATE-UI-TEXT" * 1000
    rows = [
        {
            "i": 1,
            "action": "tap",
            "args": {"x": 10, "y": 20},
            "before": {"app": secret, "activity": None, "elements": []},
            "note": secret,
        },
        {"i": 2, "action": "input_text", "args": {"text": secret}},
        {
            "i": 3,
            "action": "extension_call",
            "args": {"name": "vault.action", "arguments": {"password": secret}},
        },
    ]
    path = write_macro(tmp_path, rows=rows)
    result = CliRunner().invoke(macro, ["replay", str(path), "--dry-run"])
    assert result.exit_code == 0, result.exception
    assert "PRIVATE-UI-TEXT" not in result.output
    assert "1: tap x=10 y=20" in result.output
    assert "input_text [redacted]" in result.output
    assert "extension_call [redacted]" in result.output
    assert len(result.output) < 200


def test_macro_logging_uses_debug_format_when_requested(monkeypatch, capsys):

    # Isolate changes to the shared named logger from other CLI tests.
    logger = logging.getLogger("mobilerun-macro")
    level = logger.level
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", False)
    try:
        configured = configure_macro_logging(debug=True)
        assert configured.level == logging.DEBUG
        configured.info("hello")
        assert capsys.readouterr().err == "INFO mobilerun-macro hello\n"
    finally:
        logger.setLevel(level)
