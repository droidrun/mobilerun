"""Literal regressions from the independent CLI runtime verification."""

import importlib
import json
import logging
import os
import select
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import click
import httpx
import pytest
from click.testing import CliRunner

from mobilerun.cli import device_lifecycle, deviceauth, options
from mobilerun.cli.main import cli
from mobilerun.config_manager import ConfigLoader


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(ConfigLoader, "get_user_config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr(deviceauth, "load_credentials", lambda: None)
    for name in ("MOBILERUN_CONFIG", "MOBILERUN_DEBUG", "MOBILERUN_CLOUD_API_KEY",
                 "MOBILERUN_DEVICE", "MOBILERUN_TARGET"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("error", [
    FileNotFoundError(2, "binary missing", "/fake/adb"),
    ConnectionRefusedError("server unavailable after startup"),
])
@pytest.mark.parametrize("args", [
    ["device", "list"], ["device", "ui"], ["device", "ping"],
    ["device", "apps"], ["device", "tap", "1", "2", "-d", "serial"],
    ["device", "setup"], ["device", "connect", "127.0.0.1:5555"],
    ["device", "disconnect", "127.0.0.1:5555"],
])
def test_local_adb_missing_or_server_unreachable(monkeypatch, error, args):
    import mobilerun_core

    monkeypatch.setattr("mobilerun.cli.selection.resolve_cloud_api_key", lambda: None)
    monkeypatch.setattr("mobilerun.agent.utils.android_device.resolve_android_serial",
                        AsyncMock(side_effect=error))
    client = SimpleNamespace(list_devices=Mock(side_effect=error),
                             connect=Mock(side_effect=error), close=Mock())
    monkeypatch.setattr(mobilerun_core, "Mobilerun", lambda **kwargs: client)
    monkeypatch.setattr(device_lifecycle.adb, "connect", AsyncMock(side_effect=error))
    monkeypatch.setattr(device_lifecycle.adb, "disconnect", AsyncMock(side_effect=error))
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1, result.output
    assert result.stderr == "Error: " + options.ADB_HINT + "\n"


def test_local_adb_server_can_start_on_first_use(monkeypatch, tmp_path):
    import asyncio

    binary = tmp_path / "adb"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("ANDROID_ADB_SERVER_PORT", "5999")
    # The library owns startup: reaching list() must be allowed on a closed port.
    listing = AsyncMock(return_value=[])
    monkeypatch.setattr(device_lifecycle.adb, "list", listing)
    client = SimpleNamespace(list_devices=lambda **kwargs: asyncio.run(listing()), close=Mock())
    monkeypatch.setattr(device_lifecycle, "core_client", lambda *args, **kwargs: client)
    result = CliRunner().invoke(cli, ["device", "list"])
    assert result.exit_code == 0, result.output
    assert options.ADB_HINT not in result.output
    listing.assert_awaited_once()



def test_all_keeps_cloud_when_adb_fails(monkeypatch):
    row = SimpleNamespace(device_id="cloud-id", name="phone", state="running",
                          device_type="android", close=Mock())
    monkeypatch.setattr(device_lifecycle, "core_client", lambda *a, **k: SimpleNamespace(
        list_devices=Mock(side_effect=lambda **kwargs: [row] if kwargs["scope"] == "cloud"
                          else (_ for _ in ()).throw(ConnectionRefusedError())), close=Mock(),
    ))
    result = CliRunner().invoke(cli, ["device", "list", "--all", "-j"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["devices"][0]["device_id"] == "cloud-id"
    assert body["errors"] == [{"target": "local", "error": options.ADB_HINT}]
    assert result.stderr == "Warning: local: " + options.ADB_HINT + "\n"


@pytest.mark.parametrize("path", [["configure", "-p", "openai", "-m", "M", "-k", "-"],
                                  ["auth", "login", "anthropic", "--token", "-"]])
def test_hidden_secret_prompt_interrupt_is_130(monkeypatch, path):
    class TerminalRunner(CliRunner):
        @contextmanager
        def isolation(self, *args, **kwargs):
            with super().isolation(*args, **kwargs) as streams:
                sys.stdin.isatty = lambda: True
                yield streams

    def interrupt(*a, **k):
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt as exc:
            raise click.Abort from exc

    monkeypatch.setattr(click, "prompt", interrupt)
    result = TerminalRunner().invoke(cli, path)
    assert result.exit_code == 130, result.output
    assert "Aborted!" not in result.output


@pytest.mark.parametrize("env,flag,expected", [(None, False, True), ("false", False, False),
                                               ("false", True, True), ("true", False, True)])
def test_config_debug_precedence_and_env_agree(monkeypatch, env, flag, expected):
    ConfigLoader.get_user_config_path().write_text("_version: 7\nlogging:\n  debug: true\n")
    if env is not None:
        monkeypatch.setenv("MOBILERUN_DEBUG", env)

    def probe(params):
        logging.getLogger("mobilerun").debug("debug-policy-proof")

    monkeypatch.setattr(device_lifecycle, "probe_device", probe)
    result = CliRunner().invoke(cli, ["device", "ping", *(["-v"] if flag else [])])
    assert result.exit_code == 0, result.output
    assert ("debug-policy-proof" in result.stderr) == expected
    result = CliRunner().invoke(cli, ["env", "-j", *(["-v"] if flag else [])])
    rows = {row["variable"]: row for row in json.loads(result.stdout)}
    assert rows["MOBILERUN_DEBUG"]["value"] is expected


@pytest.mark.parametrize("flag,replacement,path", [
    ("--base_url", "-u/--llm-url", ["run"]), ("--api_base", "-u/--llm-url", ["run"]),
    ("--cloud", "-t cloud", ["run"]), ("--ios", "-P ios", ["run"]),
    ("--debug", "-v", ["run"]), ("--device-id", "-d", ["run"]),
    ("--steps", "-s/--max-steps", ["run"]), ("--path", "--apk", ["device", "setup"]),
    ("-pv", "--portal-version", ["device", "setup"]),
    ("--control-backend", "--backend", ["run"]),
])
def test_removed_flag_guidance(flag, replacement, path):
    result = CliRunner().invoke(cli, [*path, flag, "value"])
    assert result.exit_code == 2
    assert f"Removed option: {flag}. Use {replacement} instead." in result.stderr
    assert "Did you mean" not in result.output


@pytest.mark.parametrize("blocked", ["click"])
def test_startup_sigint_clean_subprocess(tmp_path, blocked):
    # Test-only import interception provides a deterministic early SIGINT window.
    # No production hook and no timing-based sleeps.
    hook = tmp_path / "sitecustomize.py"
    hook.write_text('''import builtins, sys, time
original = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == %r:
        print("READY", flush=True)
        while True:
            time.sleep(60)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked_import
''' % blocked)
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    process = subprocess.Popen([sys.executable, "-c", "from mobilerun.cli.main import cli; cli()"],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 10)[0], "startup hook not reached"
        assert process.stdout.readline() == "READY\n"
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, stderr
        assert stdout == stderr == ""
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_status_no_credential_and_logout_env(monkeypatch):
    result = CliRunner().invoke(cli, ["auth", "status"])
    assert result.exit_code == 0
    assert "cloud commands have no credential; run `mobilerun auth login` or set MOBILERUN_API_KEY" in result.stdout
    assert "use not set" not in result.stdout
    monkeypatch.setenv("MOBILERUN_CLOUD_API_KEY", "fake-env-key")
    monkeypatch.setattr(deviceauth, "clear_credentials", lambda: False)
    result = CliRunner().invoke(cli, ["auth", "logout"])
    assert result.exit_code == 0
    assert "MOBILERUN_CLOUD_API_KEY stays active; unset it" in result.stderr


def test_short_token_does_not_redact_words():
    result = CliRunner().invoke(cli, ["auth", "login", "xai", "--device-code", "--token", "x"])
    assert result.exit_code == 2
    assert "--token is not supported for xai" in result.stderr
    assert "[redacted]" not in result.output
    assert options.redact("abcdefgh xai", ["abcdefgh", "x"]) == "[redacted] xai"


def test_auth_connection_hint(monkeypatch):
    monkeypatch.setattr(deviceauth, "request_code", Mock(side_effect=httpx.ConnectError("refused")))
    result = CliRunner().invoke(cli, ["auth", "login", "--auth-url", "http://127.0.0.1:9/auth"])
    assert result.exit_code == 1
    assert "Check --auth-url or MOBILERUN_AUTH_URL: http://127.0.0.1:9/auth" in result.stderr


@pytest.mark.parametrize("args", [["device", "tap", "-5", "10"], ["device", "type", "-hello"]])
def test_leading_dash_hint(args):
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 2
    assert "Put `--` before arguments that start with a dash." in result.stderr


def test_typo_keeps_remaining_command():
    result = CliRunner().invoke(cli, ["devcie", "list"])
    assert result.exit_code == 2
    assert "Did you mean 'mobilerun device list'?" in result.stderr


def test_every_command_examples_and_verbose():
    def visit(command, path=()):
        yield command, path
        if isinstance(command, click.Group):
            for name, child in command.commands.items():
                yield from visit(child, (*path, name))

    for _, path in visit(cli):
        result = CliRunner().invoke(cli, [*path, "--help"])
        assert result.exit_code == 0
        assert "Examples:" in result.stdout, path
        assert "--verbose" in result.stdout, path


def test_readme_real_error_examples():
    readme = Path("README.md").read_text()
    assert '{"error":"Resource not found","status":404,"code":null}' in readme
    assert '`"credential":"env"`' in readme and '`"credential":"login"`' in readme
    document = json.loads(readme.split("`doctor -j`", 1)[1].split("```json\n", 1)[1].split("\n```", 1)[0])
    assert [check["name"] for check in document["checks"]] == ["SDK Version", "Config", "ADB"]
    assert document["checks"][-1]["detail"] == "No adb exe could be found. Install adb on your system"


def test_doctor_reports_library_adb_error(monkeypatch):
    import asyncio

    doctor = importlib.import_module("mobilerun.cli.doctor")
    monkeypatch.setattr(doctor.adb, "list", AsyncMock(side_effect=RuntimeError("library startup failed")))
    result = asyncio.run(doctor.check_adb(False))
    assert result.status == doctor.Status.FAIL
    assert result.message == "library startup failed"
