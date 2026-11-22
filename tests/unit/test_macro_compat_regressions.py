import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.devices.base import selection
from mobilerun.cli.devices.base.options import device_options
from mobilerun.errors.base import MacroError
from mobilerun.macro.models import normalize_ui_state
from mobilerun.macro.player import MacroPlayer
from mobilerun.macro.recorder import MacroRecorder


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
    from mobilerun.cli.devices.base import discovery

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
    from mobilerun.macro.cli.group import macro

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
    from mobilerun.macro.cli.group import macro

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
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "mobilerun.macro", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "replay" in result.stdout


def test_replay_config_reaches_device_connection(monkeypatch, tmp_path):
    import importlib

    replay_module = importlib.import_module("mobilerun.macro.cli.replay")
    from mobilerun.macro.cli.group import macro
    from mobilerun.models.results import ExecutionResult

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
    from mobilerun.macro.cli.group import macro

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
