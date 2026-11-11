"""The public CLI is a single, deliberately breaking command contract."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli import auth, device_lifecycle, deviceauth
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
    monkeypatch.setattr("mobilerun.cli.options.require_adb", lambda: "/fake/adb")
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
    assert set(cli.commands) == {"run", "device", "auth", "configure", "macro", "doctor", "env", "completion"}
    assert set(cli.commands["device"].commands) == {
        "list", "connect", "disconnect", "setup", "ping", "tap", "swipe", "long-press",
        "type", "press", "screenshot", "ui", "apps", "start", "pair",
    }
    assert set(cli.commands["auth"].commands) == {"login", "logout", "status"}
    assert "--no-qr" in {
        option for param in cli.commands["auth"].commands["login"].params
        for option in getattr(param, "opts", ())
    }
    expected = {"-d": "--device", "-t": "--target", "-P": "--platform", "-c": "--config",
                "-v": "--verbose", "-j": "--json", "-p": "--provider", "-m": "--model",
                "-u": "--llm-url", "-T": "--temperature", "-s": "--max-steps",
                "-a": "--agent", "-r": "--reasoning", "-n": "--dry-run", "-k": "--api-key", "-V": "--version"}
    for path, command in commands():
        for param in command.params:
            if isinstance(param, click.Option):
                for short in (opt for opt in param.opts if opt.startswith("-") and not opt.startswith("--")):
                    assert expected[short] in param.opts, (path, param.opts)


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
def test_env_binding_and_flag_precedence(monkeypatch, tmp_path, envvar, flag, value, override, keyword):
    if keyword == "config_path":
        value, override = str(tmp_path / value), str(tmp_path / override)
        Path(value).write_text("_version: 7\n")
        Path(override).write_text("_version: 7\n")
    callback = AsyncMock(return_value=(True, "done", 1))
    monkeypatch.setattr(run_module, "_run_task", callback)
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
    for name in ("MOBILERUN_CLOUD_API_KEY", "MOBILERUN_ANDROID_PORTAL_TOKEN", "OPENAI_API_KEY"):
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
    assert values["MOBILERUN_ANDROID_PORTAL_TOKEN"]["value"] == "set"
    assert "never-print-this-token" not in result.output
    monkeypatch.setenv("MOBILERUN_DEVICE", "from-env")
    values = {row["variable"]: row for row in json.loads(CliRunner().invoke(cli, ["env", "-j"]).stdout)}
    assert values["MOBILERUN_DEVICE"]["value"] == "from-env"
    assert values["MOBILERUN_DEVICE"]["source"] == "env"


def test_cloud_target_without_device_does_not_touch_adb(monkeypatch):
    monkeypatch.setattr("mobilerun.agent.utils.android_device.resolve_android_serial", AsyncMock(side_effect=AssertionError("must not discover")))
    result = CliRunner().invoke(cli, ["device", "ui", "-t", "cloud"])
    assert result.exit_code == 1
    assert "mobilerun device list -t cloud" in result.output


@pytest.mark.parametrize("args", [["device", "list", "-j"], ["device", "apps", "-j"],
                                  ["device", "screenshot", "-j"], ["device", "ui", "-j"]])
def test_read_commands_emit_one_document(monkeypatch, args):
    import mobilerun_core

    core = SimpleNamespace(device_id="id", capabilities={"backend": "local-android-adb"}, _conn=object())
    monkeypatch.setattr(mobilerun_core.Mobilerun, "list_devices", lambda *a, **kw: [core])
    driver = Mock()
    driver.screenshot.return_value = "cG5n"
    driver.list_apps.return_value = [{"package": "example"}]
    driver.ui_json.return_value = '{"a11y_tree": {}}'
    monkeypatch.setattr(mobilerun_core.Mobilerun, "connect", lambda *a, **kw: driver)
    args = [*args, "-d", "serial"] if "list" not in args else args
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
    from mobilerun.cli.doctor import CheckResult, Status
    monkeypatch.setattr("mobilerun.cli.doctor.run_doctor", AsyncMock(return_value=[CheckResult("Device", Status.FAIL, "missing device")]))
    result = CliRunner().invoke(cli, ["doctor", "-j"])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "ok": False, "checks": [{"name": "Device", "status": "fail", "detail": "missing device"}],
    }


@pytest.mark.parametrize("service,option", [("mobilerun", "--model"), ("xai", "--callback-host"),
                                            ("openai", "--token"), ("anthropic", "--model")])
def test_auth_rejects_unsupported_options(service, option):
    result = CliRunner().invoke(cli, ["auth", "login", service, option, "value"])
    assert result.exit_code == 2, result.output
    assert "not supported" in result.output


def test_provider_table_matches_wizard_and_loaders():
    from mobilerun.agent.providers.registry import resolve_provider_variant

    for provider in PROVIDERS.values():
        if provider.family:
            assert resolve_provider_variant(provider.family, provider.auth_mode).id == provider.runtime
    assert llm_url_kwargs("Ollama", "http://x") == {"base_url": "http://x"}
    assert llm_url_kwargs("OpenAIResponses", "http://x") == {"api_base": "http://x"}
    assert llm_url_kwargs("GoogleGenAI", "http://x") == {"http_options": {"base_url": "http://x"}}


def test_interrupt_exits_130(monkeypatch):
    monkeypatch.setattr(device_lifecycle.adb, "connect", AsyncMock(side_effect=KeyboardInterrupt))
    result = CliRunner().invoke(cli, ["device", "connect", "serial"])
    assert result.exit_code == 130


def test_import_has_no_cli_global_side_effects():
    script = """
import os
import warnings
os.environ.pop("TOKENIZERS_PARALLELISM", None)
os.environ.pop("GRPC_ENABLE_FORK_SUPPORT", None)
original = dict(os.environ)
import mobilerun.cli.main
assert dict(os.environ) == original
assert not any(action == 'ignore' and message is None and category is Warning
               and module is None for action, message, category, module, *_ in warnings.filters)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def fake_agent(result, calls=None):
    class Handler:
        async def stream_events(self):
            print("progress on stderr")
            if False:
                yield None
        def __await__(self):
            async def done():
                return result
            return done().__await__()
    class Agent:
        def __init__(self, **kwargs):
            if calls is not None:
                calls.append(kwargs)
        def run(self):
            return Handler()
    return Agent


@pytest.mark.parametrize("success", [True, False])
def test_run_json_is_built_from_result_event(monkeypatch, success):
    monkeypatch.setattr("mobilerun.MobileAgent", fake_agent(SimpleNamespace(success=success, reason="observed result", steps=4)))
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
    monkeypatch.setattr(cloud, "CloudDriver", Driver)
    monkeypatch.setattr("mobilerun.MobileAgent", fake_agent(SimpleNamespace(success=True, reason="done", steps=1), calls))
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
    assert "session check failed" in json.loads(result.stdout)["error"]
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

    monkeypatch.setattr("mobilerun.macro.replay.MacroPlayer", Player)
    monkeypatch.setattr(macro, "select_device", AsyncMock(return_value=SimpleNamespace(device="serial", config=MobileConfig())))
    monkeypatch.setattr(macro, "_native_driver", AsyncMock(return_value=object()))
    result = CliRunner().invoke(cli, ["macro", "replay", str(path)])
    assert result.exit_code == 1
    assert "Macro replay failed" in result.output


@pytest.mark.parametrize("stored_url", ["https://issuer.test/api/auth", None])
def test_status_uses_only_credential_issuer(monkeypatch, stored_url):
    monkeypatch.setenv("MOBILERUN_AUTH_URL", "http://other.test/api/auth")
    monkeypatch.setattr(deviceauth, "load_credentials", lambda: {"access_token": "marker-token", "auth_url": stored_url})
    fetch = Mock(return_value={"user": {"email": "user@example.test"}})
    monkeypatch.setattr(deviceauth, "fetch_session", fetch)
    result = CliRunner().invoke(cli, ["auth", "status", "-j"])
    assert result.exit_code == 0, result.output
    fetch.assert_called_once_with(stored_url or auth.DEFAULT_AUTH_URL, "marker-token")
    assert CliRunner().invoke(cli, ["auth", "status", "--auth-url", "http://other.test"]).exit_code == 2


@pytest.mark.parametrize("verb", ["connect", "disconnect"])
def test_connection_requires_positional_and_ignores_device_env(monkeypatch, verb):
    monkeypatch.setenv("MOBILERUN_DEVICE", "wrong-device")
    monkeypatch.setenv("MOBILERUN_TARGET", "cloud")
    call = AsyncMock(return_value="connected to correct-device" if verb == "connect" else True)
    monkeypatch.setattr(device_lifecycle.adb, verb, call)
    runner = CliRunner()
    assert runner.invoke(cli, ["device", verb]).exit_code == 2
    result = runner.invoke(cli, ["device", verb, "correct-device"])
    assert result.exit_code == 0, result.output
    assert call.await_args.args == ("correct-device",)
    for option in ("--device", "--target", "--platform", "--config", "--api-url", "--tcp"):
        assert runner.invoke(cli, ["device", verb, "serial", option, "x"]).exit_code == 2


@pytest.mark.parametrize("scope", ["cloud", "all"])
def test_cloud_list_includes_every_state_and_metadata(monkeypatch, scope):
    import mobilerun_core

    monkeypatch.setenv("MOBILERUN_CLOUD_API_KEY", "fake-key")
    rows = [SimpleNamespace(device_id=f"id-{state}", _conn=SimpleNamespace(
        _device_cache=SimpleNamespace(name=f"device-{state}", state=state, type="android")))
        for state in ("ready", "stopped", "provisioning")]
    list_call = Mock(side_effect=lambda *a, **kw: rows if kw["scope"] == "cloud" else [])
    monkeypatch.setattr(mobilerun_core.Mobilerun, "list_devices", list_call)
    flags = ["--all"] if scope == "all" else ["-t", "cloud"]
    result = CliRunner().invoke(cli, ["device", "list", *flags, "-j"])
    assert result.exit_code == 0, result.output
    assert [call.kwargs for call in list_call.call_args_list] == [
        {"filters": {"state": None}, "scope": target}
        for target in (["local", "cloud"] if scope == "all" else ["cloud"])
    ]
    data = json.loads(result.stdout)
    if scope == "all":
        assert data["errors"] == []
        data = data["devices"]
    assert {row["state"] for row in data} == {"ready", "stopped", "provisioning"}
    assert all(set(row) == {"device_id", "name", "state", "type"} for row in data)
    text = CliRunner().invoke(cli, ["device", "list", *flags])
    assert all(f"id-{state}\tdevice-{state}\t{state}\tandroid" in text.stdout for state in ("ready", "stopped", "provisioning"))


@pytest.mark.parametrize("runtime,expected", [("DeepSeek", {"base_url": "http://x"}),
                                              ("xai_oauth", None), ("CustomProvider", None)])
def test_config_provider_url_never_crashes_with_stop_iteration(monkeypatch, runtime, expected):
    config = MobileConfig()
    for profile in config.llm_profiles.values():
        profile.provider = runtime
    monkeypatch.setattr(run_module.ConfigLoader, "load", lambda _: config)
    monkeypatch.setattr("mobilerun.MobileAgent", fake_agent(SimpleNamespace(success=True, reason="done", steps=1)))
    result = CliRunner().invoke(cli, ["run", "task", "-u", "http://x", "-j"])
    assert "StopIteration" not in result.output
    if expected:
        assert result.exit_code == 0, result.output
        assert all(profile.kwargs["base_url"] == "http://x" for profile in config.llm_profiles.values())
    else:
        assert result.exit_code == 2, result.output
        assert result.stdout == ""
        assert "not supported for provider" in result.stderr


@pytest.mark.parametrize("provider", ["openrouter", "deepseek"])
def test_run_only_providers_work_and_configure_rejects_them(monkeypatch, provider):
    llm = object()
    load = Mock(return_value=llm)
    monkeypatch.setattr("mobilerun.agent.utils.llm_picker.load_llm", load)
    monkeypatch.setattr("mobilerun.MobileAgent", fake_agent(SimpleNamespace(success=True, reason="done", steps=1)))
    result = CliRunner().invoke(cli, ["run", "task", "-p", provider, "-m", "test-model", "-j"])
    assert result.exit_code == 0, result.output
    assert load.call_args.args == (PROVIDERS[provider].runtime,)
    result = CliRunner().invoke(cli, ["configure", "-p", provider])
    assert result.exit_code == 2
    assert "Invalid value" in result.output


@pytest.mark.parametrize("args,verb,expected_args,expected_kwargs", [
    (["tap", "10", "20"], "tap", (10, 20), {"humanize": False}),
    (["swipe", "1", "2", "3", "4", "--duration", "0.4"], "swipe", (1, 2, 3, 4), {"ms": 400, "humanize": False}),
    (["long-press", "10", "20"], "swipe", (10, 20, 10, 20), {"ms": 1000, "humanize": False}),
    (["type", "hello", "--clear"], "type", ("hello",), {"clear": True, "humanize": False}),
    (["press", "home"], "key", ("home",), {}),
    (["start", "example.app"], "start_app", ("example.app",), {}),
    (["apps", "--system", "-j"], "list_apps", (), {"include_system_apps": True}),
    (["screenshot", "-j"], "screenshot", (), {}),
    (["ui", "-j"], "ui_json", (), {}),
    (["ping"], "ui", (), {}),
])
def test_device_verbs_call_core_directly(monkeypatch, args, verb, expected_args, expected_kwargs):
    import mobilerun_core

    driver = Mock()
    driver.list_apps.return_value = [{"package": "example"}]
    driver.screenshot.return_value = "cG5n"
    driver.ui_json.return_value = '{"a11y_tree": {"text": "Hello"}}'
    core = Mock()
    core.connect.return_value = driver
    factory = Mock(return_value=core)
    monkeypatch.setattr(mobilerun_core, "Mobilerun", factory)
    result = CliRunner().invoke(cli, ["device", *args, "-d", "serial"])
    assert result.exit_code == 0, result.output
    getattr(driver, verb).assert_called_once_with(*expected_args, **expected_kwargs)
    assert core.connect.call_args.args == ("serial",)
    assert core.connect.call_args.kwargs["backend"] == "local-android-adb"
    assert factory.call_args.kwargs["base_url"] == "https://api.mobilerun.ai/v1"
    driver.close.assert_called_once()
    core.close.assert_called_once()
    if args[0] == "screenshot":
        Path(json.loads(result.stdout)["path"]).unlink()
    if args[0] in {"ui", "apps"}:
        assert json.loads(result.stdout)


@pytest.mark.parametrize("args", [["device", "ui", "-t", "cloud"], ["device", "list", "-t", "cloud"],
                                  ["device", "apps", "-t", "cloud"], ["device", "screenshot", "-t", "cloud"],
                                  ["env"], ["run", "task"]])
def test_json_failures_emit_exactly_one_error(monkeypatch, args, tmp_path):
    if args[0] in {"env", "run"}:
        monkeypatch.setenv("MOBILERUN_CONFIG", str(tmp_path / "missing.yaml"))
    result = CliRunner().invoke(cli, [*args, "-j"])
    assert result.exit_code == 1, result.output
    document = json.loads(result.stdout)
    assert set(document) == {"error"}
    assert document["error"]


def test_unsupported_core_operation_includes_alternative(monkeypatch):
    import mobilerun_core
    from mobilerun_core.connection._unsupported import UnsupportedOperation

    driver = Mock()
    driver.start_app.side_effect = UnsupportedOperation(verb="start_app", backend="local-ios-http", reason="unavailable", alternative="open_deep_link")
    monkeypatch.setattr(mobilerun_core.Mobilerun, "connect", lambda *a, **kw: driver)
    result = CliRunner().invoke(cli, ["device", "start", "example", "-d", "serial"])
    assert result.exit_code == 1
    assert "unavailable" in result.stderr and "open_deep_link" in result.stderr


def test_missing_portal_token_is_cli_error_before_core(monkeypatch):
    import mobilerun_core
    connect = Mock()
    monkeypatch.setattr(mobilerun_core.Mobilerun, "connect", connect)
    monkeypatch.delenv("MOBILERUN_ANDROID_PORTAL_TOKEN", raising=False)
    result = CliRunner().invoke(cli, ["device", "tap", "1", "2", "-d", "http://127.0.0.1:9"])
    assert result.exit_code == 1
    assert "MOBILERUN_ANDROID_PORTAL_TOKEN" in result.stderr
    connect.assert_not_called()


def test_env_path_and_unset_sources_and_effective_telemetry(monkeypatch, tmp_path):
    config = MobileConfig()
    config.telemetry.enabled = False
    cfg = tmp_path / "custom.yaml"
    ConfigLoader.save(config, path=cfg)
    monkeypatch.setenv("MOBILERUN_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("MOBILERUN_CONFIG", str(cfg))
    result = CliRunner().invoke(cli, ["env", "-j"])
    rows = {row["variable"]: row for row in json.loads(result.stdout)}
    assert rows["MOBILERUN_CONFIG"]["source"] == "env"
    assert rows["MOBILERUN_DEVICE"]["source"] == "not set"
    assert rows["MOBILERUN_ANDROID_PORTAL_TOKEN"]["source"] == "not set"
    assert rows["MOBILERUN_TELEMETRY_ENABLED"]["value"] is False
    other = tmp_path / "other.yaml"
    ConfigLoader.save(config, path=other)
    result = CliRunner().invoke(cli, ["env", "-j", "-c", str(other)])
    row = next(row for row in json.loads(result.stdout) if row["variable"] == "MOBILERUN_CONFIG")
    assert row["value"] == str(other) and row["source"] == "flag"


@pytest.mark.parametrize("args", [["env"], ["run", "task"], ["device", "ui"], ["device", "list"], ["auth", "status"], ["doctor"]])
def test_missing_env_config_is_an_error(monkeypatch, tmp_path, args):
    from mobilerun.cli import doctor
    from mobilerun.cli.doctor import CheckResult, Status
    monkeypatch.setenv("MOBILERUN_CONFIG", str(tmp_path / "missing.yaml"))
    monkeypatch.setattr(doctor, "check_sdk_version", AsyncMock(return_value=CheckResult("SDK", Status.PASS, "ok")))
    monkeypatch.setattr(doctor, "check_adb", AsyncMock(return_value=CheckResult("ADB", Status.FAIL, "not installed")))
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1, result.output
    assert "Config" in result.output or "config" in result.output


def test_doctor_checks_system_before_ambiguous_device(monkeypatch):
    from mobilerun.cli import doctor
    from mobilerun.cli.doctor import CheckResult, Status
    order = []
    async def sdk(debug):
        order.append("sdk")
        return CheckResult("SDK", Status.PASS, "ok")
    async def adb_check(debug):
        order.append("adb")
        return CheckResult("ADB", Status.PASS, "ok")
    async def device_check(serial, debug):
        order.append("device")
        return CheckResult("Device", Status.FAIL, "multiple devices"), None
    monkeypatch.setattr(doctor, "check_sdk_version", sdk)
    monkeypatch.setattr(doctor, "check_adb", adb_check)
    monkeypatch.setattr(doctor, "check_device", device_check)
    result = CliRunner().invoke(cli, ["doctor"])
    assert result.exit_code == 1, result.output
    assert order == ["sdk", "adb", "device"]
    assert all(name in result.stdout for name in ("SDK", "Config", "ADB", "Device"))


def test_configure_api_provider_preselects_api_key_and_validates_auth_mode(monkeypatch):
    configure = importlib.import_module("mobilerun.cli.configure")
    wizard = Mock()
    monkeypatch.setattr("mobilerun.cli.configure_wizard.run_configure_wizard", wizard)
    monkeypatch.setattr(configure, "input_allowed", lambda _: True)
    result = CliRunner().invoke(cli, ["configure", "-p", "openai"])
    assert result.exit_code == 0, result.output
    assert wizard.call_args.kwargs["auth_mode"] == "api_key"
    assert CliRunner().invoke(cli, ["configure", "--auth-mode", "invalid"]).exit_code == 2


def test_run_help_task_and_agent_validation():
    result = CliRunner().invoke(cli, ["run", "-h"])
    assert "TASK" in result.stdout
    assert CliRunner().invoke(cli, ["run", "task", "-a", "no-such-agent"]).exit_code == 2


def test_macro_replay_unexpected_error_printed_once_with_traceback(monkeypatch, tmp_path):
    class Player:
        def __init__(self, **kwargs):
            pass
        def load_macro_from_file(self, path):
            raise RuntimeError("unique macro failure")
    path = tmp_path / "macro.json"
    path.write_text("{}")
    monkeypatch.setattr("mobilerun.macro.replay.MacroPlayer", Player)
    result = CliRunner().invoke(cli, ["macro", "replay", str(path), "-n"])
    assert result.exit_code == 1
    assert result.stderr.count("unique macro failure") == 1
    verbose = CliRunner().invoke(cli, ["macro", "replay", str(path), "-n", "-v"])
    assert verbose.exit_code == 1
    assert "Traceback" in verbose.stderr


def test_device_commands_do_not_accept_tcp_or_unused_list_flags():
    runner = CliRunner()
    for path, command in commands(cli.commands["device"], ("device",)):
        assert all("--tcp" not in getattr(param, "opts", ()) for param in command.params), path
    for flag in ("-a", "-d", "-P", "-c"):
        assert runner.invoke(cli, ["device", "list", flag, "x"]).exit_code == 2
