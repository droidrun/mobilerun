"""The public CLI is a single, deliberately breaking command contract."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli import auth, device_commands, device_lifecycle, deviceauth, selection
from mobilerun.cli.main import cli
from mobilerun.cli.providers import PROVIDERS, llm_url_kwargs
from mobilerun.config_manager import ConfigLoader, MobileConfig, credential_paths

run_module = importlib.import_module("mobilerun.cli.run")


def commands(group=cli, path=()):
    yield path, group
    if isinstance(group, click.Group):
        for name, command in group.commands.items():
            yield from commands(command, (*path, name))


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr(ConfigLoader, "get_user_config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr(credential_paths, "OAUTH_CREDENTIAL_DIR", tmp_path / "credentials")
    monkeypatch.setattr(credential_paths, "AUTH_PROFILES_PATH", tmp_path / "auth-profiles.json")
    monkeypatch.setattr(auth, "SERVICES", {
        service: (tmp_path / "auth-profiles.json", variant)
        for service, (_, variant) in auth.SERVICES.items()
    })
    for name in ("MOBILERUN_DEVICE", "MOBILERUN_CONFIG", "MOBILERUN_TARGET", "MOBILERUN_DEBUG",
                 "MOBILERUN_API_BASE_URL", "MOBILERUN_CLOUD_API_KEY", "MOBILERUN_AUTH_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("path,command", list(commands()))
@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_every_command_has_help(path, command, flag):
    result = CliRunner().invoke(cli, [*path, flag])
    assert result.exit_code == 0, result.output


def test_tree_and_short_options():
    assert set(cli.commands) == {"run", "device", "auth", "configure", "macro", "doctor", "env"}
    assert set(cli.commands["device"].commands) == {
        "list", "connect", "disconnect", "setup", "ping", "tap", "swipe", "long-press",
        "type", "press", "screenshot", "ui", "apps", "start",
    }
    assert set(cli.commands["auth"].commands) == {"login", "logout", "status"}
    expected = {"-d": "--device", "-t": "--target", "-P": "--platform", "-c": "--config",
                "-v": "--verbose", "-j": "--json", "-p": "--provider", "-m": "--model",
                "-u": "--llm-url", "-T": "--temperature", "-s": "--max-steps",
                "-r": "--reasoning", "-n": "--dry-run", "-k": "--api-key", "-V": "--version"}
    for path, command in commands():
        for param in command.params:
            if isinstance(param, click.Option):
                for short in (opt for opt in param.opts if opt.startswith("-") and not opt.startswith("--")):
                    # The frozen command-specific contract explicitly assigns -a twice.
                    long = "--all" if path == ("device", "list") else "--agent"
                    assert (expected.get(short, long)) in param.opts, (path, param.opts)


@pytest.mark.parametrize("args", [
    ["devices"], ["connect", "serial"], ["disconnect", "serial"], ["setup"], ["ping"],
    ["login"], ["logout"], ["whoami"], ["configure", "openai"],
    ["configure", "anthropic"], ["configure", "gemini"], ["configure", "xai"],
    ["run", "x", "--base_url", "http://x"], ["run", "x", "--api_base", "http://x"],
    ["run", "x", "--steps", "1"], ["run", "x", "--debug"], ["run", "x", "--ios"],
    ["run", "x", "--no-vision-only"], ["run", "x", "--control-backend", "visual-remote"],
    ["run", "x", "--device-id", "x"], ["device", "ui", "--cloud"],
    ["device", "ui", "--device-id", "x"], ["device", "ui", "--base-url", "http://x"],
    ["device", "setup", "--path", "x"], ["device", "setup", "-pv", "x"],
])
def test_old_commands_and_flags_are_usage_errors(args):
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 2, result.output


@pytest.mark.parametrize("envvar,flag,value,override,keyword", [
    ("MOBILERUN_DEVICE", "--device", "serial-a", "serial-b", "device"),
    ("MOBILERUN_TARGET", "--target", "cloud", "local", "target"),
    ("MOBILERUN_API_BASE_URL", "--api-url", "https://a.test", "https://b.test", "api_url"),
    ("MOBILERUN_CONFIG", "--config", "a.yaml", "b.yaml", "config_path"),
    ("MOBILERUN_DEBUG", "--verbose", "false", None, "debug"),
])
def test_env_binding_and_flag_precedence(monkeypatch, envvar, flag, value, override, keyword):
    callback = AsyncMock(return_value=True)
    monkeypatch.setattr(run_module, "run_command", callback)
    monkeypatch.setenv(envvar, value)
    result = CliRunner().invoke(cli, ["run", "task"])
    assert result.exit_code == 0, result.output
    assert callback.await_args.kwargs[keyword] == (False if keyword == "debug" else value)
    args = ["run", "task", flag] + ([override] if override is not None else [])
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert callback.await_args.kwargs[keyword] == (True if keyword == "debug" else override)
    help_result = CliRunner().invoke(cli, ["run", "--help"])
    assert envvar in help_result.output


def test_auth_url_env_and_flag_precedence(monkeypatch):
    calls = []
    monkeypatch.setattr(auth, "_login_mobilerun", lambda *args: calls.append(args))
    monkeypatch.setenv("MOBILERUN_AUTH_URL", "https://a.test")
    runner = CliRunner()
    assert runner.invoke(cli, ["auth", "login"]).exit_code == 0
    assert calls[-1][0] == "https://a.test"
    assert runner.invoke(cli, ["auth", "login", "--auth-url", "https://b.test"]).exit_code == 0
    assert calls[-1][0] == "https://b.test"


@pytest.mark.parametrize("args", [["env", "-j"], ["auth", "status", "-j"]])
def test_secrets_are_never_printed(monkeypatch, args):
    secret = "contract-secret-marker"
    for name in ("MOBILERUN_CLOUD_API_KEY", "MOBILERUN_DEVICE_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, secret)
    monkeypatch.setattr(deviceauth, "load_credentials", lambda: {"access_token": secret})
    monkeypatch.setattr(deviceauth, "fetch_session", lambda *_: {"user": {"email": "example@example.test"}})
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert secret not in result.output
    assert isinstance(json.loads(result.stdout), list)


def test_env_sources_and_secret_config(monkeypatch, tmp_path):
    config = MobileConfig()
    config.device.serial = "from-config"
    config.device.auth_token = "never-print-this-token"
    ConfigLoader.save(config)
    result = CliRunner().invoke(cli, ["env", "-j"])
    values = {row["variable"]: row for row in json.loads(result.stdout)}
    assert values["MOBILERUN_DEVICE"] == {"variable": "MOBILERUN_DEVICE", "value": "from-config", "source": "config"}
    assert values["MOBILERUN_DEVICE_TOKEN"]["value"] == "set"
    assert "never-print-this-token" not in result.output
    monkeypatch.setenv("MOBILERUN_DEVICE", "from-env")
    values = {row["variable"]: row for row in json.loads(CliRunner().invoke(cli, ["env", "-j"]).stdout)}
    assert values["MOBILERUN_DEVICE"]["value"] == "from-env"
    assert values["MOBILERUN_DEVICE"]["source"] == "env"


def test_cloud_target_without_device_does_not_touch_adb(monkeypatch):
    monkeypatch.setattr(selection, "resolve_android_serial", AsyncMock(side_effect=AssertionError("must not discover")))
    result = CliRunner().invoke(cli, ["device", "ui", "-t", "cloud"])
    assert result.exit_code == 1
    assert "mobilerun device list -t cloud" in result.output


@pytest.mark.parametrize("args", [["device", "list", "-j"], ["device", "apps", "-j"],
                                  ["device", "screenshot", "-j"], ["device", "ui", "-j"]])
def test_read_commands_emit_one_document(monkeypatch, args):
    import mobilerun_core

    core = SimpleNamespace(device_id="id", capabilities={})
    monkeypatch.setattr(mobilerun_core.Mobilerun, "list_devices", lambda *a, **kw: [core])
    driver = SimpleNamespace(screenshot=AsyncMock(return_value=b"png"),
                             get_apps=AsyncMock(return_value=[{"package": "example"}]))
    monkeypatch.setattr(device_commands, "connect_driver", AsyncMock(return_value=(driver, False)))
    provider = SimpleNamespace(get_state=AsyncMock(return_value=SimpleNamespace(formatted_text="tree", phone_state={})))
    monkeypatch.setattr(device_commands, "AndroidStateProvider", lambda *a, **kw: provider)
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    if "screenshot" in args:
        Path(document["path"]).unlink()


def test_macro_list_json(tmp_path):
    (tmp_path / "empty").mkdir()
    result = CliRunner().invoke(cli, ["macro", "list", str(tmp_path), "-j"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []


def test_doctor_failure_json(monkeypatch):
    doctor = importlib.import_module("mobilerun.cli.doctor_command")
    monkeypatch.setattr(doctor, "select_device", AsyncMock(side_effect=RuntimeError("missing device")))
    result = CliRunner().invoke(cli, ["doctor", "-j"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)[0]["status"] == "fail"


@pytest.mark.parametrize("service,option", [("mobilerun", "--model"), ("xai", "--callback-host"),
                                            ("openai", "--token"), ("anthropic", "--model")])
def test_auth_rejects_unsupported_options(service, option):
    result = CliRunner().invoke(cli, ["auth", "login", service, option, "value"])
    assert result.exit_code == 2, result.output
    assert "not supported" in result.output


def test_provider_table_matches_wizard_and_loaders():
    from mobilerun.agent.providers.registry import resolve_provider_variant

    for provider in PROVIDERS.values():
        assert resolve_provider_variant(provider.family, provider.auth_mode).id == provider.runtime
    assert llm_url_kwargs("Ollama", "http://x") == {"base_url": "http://x"}
    assert llm_url_kwargs("OpenAIResponses", "http://x") == {"api_base": "http://x"}
    assert llm_url_kwargs("GoogleGenAI", "http://x") == {"http_options": {"base_url": "http://x"}}


def test_interrupt_exits_130(monkeypatch):
    monkeypatch.setattr(device_lifecycle.adb, "connect", AsyncMock(side_effect=KeyboardInterrupt))
    result = CliRunner().invoke(cli, ["device", "connect", "serial"])
    assert result.exit_code == 130


def test_import_has_no_cli_global_side_effects(monkeypatch):
    import warnings

    import mobilerun.cli.main as main

    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    monkeypatch.delenv("GRPC_ENABLE_FORK_SUPPORT", raising=False)
    original = list(warnings.filters)
    importlib.reload(main)
    import os
    assert "TOKENIZERS_PARALLELISM" not in os.environ
    assert "GRPC_ENABLE_FORK_SUPPORT" not in os.environ
    assert warnings.filters == original


@pytest.mark.parametrize("success", [True, False])
def test_run_json_is_built_from_result_event(monkeypatch, success):
    class Handler:
        async def stream_events(self):
            print("progress on stderr")
            if False:
                yield None

        def __await__(self):
            async def done():
                return SimpleNamespace(success=success, reason="observed result", steps=4)
            return done().__await__()

    class Agent:
        def __init__(self, **kwargs):
            pass
        def run(self):
            return Handler()

    monkeypatch.setattr(run_module, "MobileAgent", Agent)
    result = CliRunner().invoke(cli, ["run", "task", "-j"])
    assert result.exit_code == (0 if success else 1), result.output
    assert json.loads(result.stdout) == {"success": success, "reason": "observed result", "steps": 4}
    assert "progress on stderr" in result.stderr
    assert "Try Mobilerun Cloud" in result.stderr


def test_run_injects_cloud_driver_without_agent_changes(monkeypatch):
    from mobilerun_core_local.driver import cloud

    calls = []
    class Driver:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        async def connect(self):
            pass
    class Agent:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def run(self):
            class Handler:
                async def stream_events(self):
                    if False:
                        yield None
                def __await__(self):
                    async def done():
                        return SimpleNamespace(success=True, reason="done", steps=1)
                    return done().__await__()
            return Handler()
    monkeypatch.setattr(cloud, "CloudDriver", Driver)
    monkeypatch.setattr(run_module, "MobileAgent", Agent)
    monkeypatch.setenv("MOBILERUN_CLOUD_API_KEY", "fake-key")
    device_id = "123e4567-e89b-12d3-a456-426614174000"
    result = CliRunner().invoke(cli, ["run", "task", "-d", device_id, "-j", "--api-url", "https://example.test/v1"])
    assert result.exit_code == 0, result.output
    assert calls[0] == {"device_id": device_id, "api_key": "fake-key", "base_url": "https://example.test/v1"}
    assert isinstance(calls[1]["driver"], Driver)
    assert calls[1]["config"].device.serial is None
    assert calls[1]["config"].device.auto_setup is False
    assert json.loads(result.stdout)["success"] is True


def test_auth_logout_preserves_other_services(tmp_path):
    path = tmp_path / "auth-profiles.json"
    path.write_text(json.dumps({"openaiOauth": {"access": "fake-openai"},
                                "grokOauth": {"access_token": "fake-xai"},
                                "apiKeys": {"openai": "fake-key"}}))
    result = CliRunner().invoke(cli, ["auth", "logout", "openai"])
    assert result.exit_code == 0, result.output
    data = json.loads(path.read_text())
    assert "openaiOauth" not in data
    assert data["grokOauth"] == {"access_token": "fake-xai"}
    assert data["apiKeys"] == {"openai": "fake-key"}
    result = CliRunner().invoke(cli, ["auth", "logout", "openai"])
    assert result.exit_code == 0
    assert "Not logged in" in result.output


def test_auth_status_session_error_prints_json_without_token(monkeypatch):
    monkeypatch.setattr(deviceauth, "load_credentials", lambda: {"access_token": "fake-secret"})
    def fail(*args):
        raise RuntimeError("fake-secret must never be printed")
    monkeypatch.setattr(deviceauth, "fetch_session", fail)
    result = CliRunner().invoke(cli, ["auth", "status", "-j"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)[0]["error"] == "session check failed"
    assert "fake-secret" not in result.output


def test_failed_macro_replay_exits_nonzero(monkeypatch, tmp_path):
    macro = importlib.import_module("mobilerun.macro.cli")
    path = tmp_path / "macro.json"
    path.write_text('{}')

    class Player:
        def __init__(self, **kwargs):
            pass
        def load_macro_from_file(self, path):
            return {"description": "failing macro", "actions": [], "total_actions": 0}
        async def replay_macro(self, *args, **kwargs):
            return False

    monkeypatch.setattr(macro, "MacroPlayer", Player)
    monkeypatch.setattr(macro, "select_device", AsyncMock(return_value=SimpleNamespace(device="serial")))
    monkeypatch.setattr(device_commands, "connect_driver", AsyncMock(return_value=(object(), False)))
    result = CliRunner().invoke(cli, ["macro", "replay", str(path)])
    assert result.exit_code == 1
    assert "Macro replay failed" in result.output
