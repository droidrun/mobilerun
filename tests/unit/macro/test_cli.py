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
from mobilerun.macro.errors import MacroError
from mobilerun.macro.models import normalize_ui_state
from mobilerun.macro.player import MacroPlayer
from mobilerun.macro.recorder import MacroRecorder
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
    assert "mismatch before step 1" in result.output
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


def test_record_without_state_has_no_guard():
    action = MacroRecorder().record_wait(0)
    assert action["pre_state"] is None
    assert action.get("post_state") is None
    assert asyncio.run(MacroPlayer().wait_for_pre_state(action["pre_state"], 1))[0]


def test_duck_typed_state_can_omit_optional_attributes():
    state = normalize_ui_state(SimpleNamespace())
    assert state["nodes"] == []
    assert state["screen"] == {"width": None, "height": None}


def test_unknown_keycode_is_error():
    with pytest.raises(MacroError, match="keycode"):
        asyncio.run(
            MacroPlayer(device=object()).replay_action(
                {"type": "key_press", "keycode": 999}
            )
        )


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
    path = tmp_path / "macro.json"
    path.write_text(
        json.dumps({"macro_schema_version": "2.0", "actions": [{"type": "wait"}]})
    )
    result = CliRunner().invoke(macro, ["replay", str(path), "--dry-run"])
    assert result.exit_code == 0, repr(result.exception)
    assert "wait" in result.output
    discovery.assert_not_called()


def test_legacy_replay_flags_are_accepted(monkeypatch, tmp_path):

    path = tmp_path / "macro.json"
    path.write_text(json.dumps({"macro_schema_version": "2.0", "actions": []}))
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
    path = tmp_path / "macro.json"
    path.write_text(json.dumps({"macro_schema_version": "2.0", "actions": []}))
    result = CliRunner().invoke(macro, ["replay", str(path), "-d", "serial"])
    assert result.exit_code == 0, repr(result.exception)
    assert connect.await_args.kwargs["config"] is config


def test_macro_agent_handoff_accepts_injected_runner(tmp_path):

    path = tmp_path / "macro.json"
    path.write_text("{}")
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
    actions = [
        {
            "action_type": "tap",
            "x": 10,
            "y": 20,
            "pre_state": {"text": secret},
            "post_state": {"text": secret},
        },
        {"type": "input_text", "text": secret},
        {"type": "extension_call", "arguments": {"password": secret}},
    ]
    path = tmp_path / "macro.json"
    path.write_text(json.dumps({"macro_schema_version": "2.0", "actions": actions}))
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
