"""Regressions for CLI discovery, config refusal and early run validation."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from async_adbutils import AdbTimeout
from click.testing import CliRunner

from mobilerun.cli import device_lifecycle, deviceauth, selection
from mobilerun.cli.main import cli
from mobilerun.cli.options import ADB_HINT
from mobilerun.config_manager import ConfigLoader, MobileConfig


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ConfigLoader, "get_user_config_path", lambda: tmp_path / "config.yaml"
    )
    monkeypatch.setattr(deviceauth, "load_credentials", lambda: None)
    monkeypatch.setenv("MOBILERUN_TELEMETRY_ENABLED", "false")
    for name in (
        "MOBILERUN_CONFIG",
        "MOBILERUN_DEVICE",
        "MOBILERUN_TARGET",
        "MOBILERUN_DEBUG",
        "ANDROID_SERIAL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("No module named 'adbutils'", name="adbutils"),
        ModuleNotFoundError(
            "No module named 'adbutils.binaries'", name="adbutils.binaries"
        ),
    ],
)
@pytest.mark.parametrize(
    "args",
    [
        ["device", "list"],
        ["device", "list", "-j"],
        ["device", "ui"],
        ["device", "ping"],
        ["device", "apps"],
        ["device", "setup"],
        ["doctor"],
        ["doctor", "-j"],
    ],
)
def test_adb_startup_failures_have_one_shared_hint(monkeypatch, error, args):
    doctor = importlib.import_module("mobilerun.cli.doctor")
    monkeypatch.setattr(
        doctor,
        "check_sdk_version",
        AsyncMock(
            return_value=doctor.CheckResult("SDK Version", doctor.Status.PASS, "test")
        ),
    )
    monkeypatch.setattr(device_lifecycle.adb, "list", AsyncMock(side_effect=error))
    # Core swallows ADB discovery failures and returns an empty local listing.
    client = SimpleNamespace(list_devices=Mock(return_value=[]), close=Mock())
    monkeypatch.setattr(device_lifecycle, "core_client", lambda *a, **kw: client)
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1, result.output
    if args[0] == "doctor":
        if "-j" in args:
            checks = json.loads(result.stdout)["checks"]
            assert checks[-1] == {"name": "ADB", "status": "fail", "detail": ADB_HINT}
        else:
            assert "ADB: fail: " + ADB_HINT in result.stdout
    else:
        assert result.stderr == "Error: " + ADB_HINT + "\n"
        if "-j" in args:
            assert json.loads(result.stdout) == {"error": ADB_HINT}
    assert "No module named" not in result.output
    assert "No devices found" not in result.output


def test_adb_translation_does_not_hide_unrelated_missing_modules():
    error = ModuleNotFoundError("missing other dependency", name="other_dependency")
    with pytest.raises(ModuleNotFoundError) as raised, selection.local_adb_errors():
        raise error
    assert raised.value is error


def test_all_devices_keeps_cloud_after_swallowed_local_adb_failure(monkeypatch):
    monkeypatch.setattr(
        device_lifecycle.adb,
        "list",
        AsyncMock(side_effect=ModuleNotFoundError(name="adbutils")),
    )
    calls = []
    device = SimpleNamespace(
        device_id="cloud-device",
        name="phone",
        state="ready",
        device_type="android",
        close=Mock(),
    )

    def listing(**kwargs):
        calls.append(kwargs["scope"])
        return [] if kwargs["scope"] == "local" else [device]

    client = SimpleNamespace(list_devices=listing, close=Mock())
    monkeypatch.setattr(device_lifecycle, "core_client", lambda *a, **kw: client)
    result = CliRunner().invoke(cli, ["device", "list", "--all", "-j"])
    assert result.exit_code == 0, result.output
    assert calls == ["local", "cloud"]
    document = json.loads(result.stdout)
    assert document["devices"][0]["device_id"] == "cloud-device"
    assert document["errors"] == [{"target": "local", "error": ADB_HINT}]
    assert result.stderr == "Warning: local: " + ADB_HINT + "\n"
    assert client.close.call_count == 2
    device.close.assert_called_once()


@pytest.mark.parametrize("command", [["device", "list"], ["macro", "list"]])
@pytest.mark.parametrize("via_env", [False, True])
def test_listing_config_directory_is_a_shared_usage_error(
    monkeypatch, tmp_path, command, via_env
):
    args = [*command, *([str(tmp_path)] if command[0] == "macro" else [])]
    if via_env:
        monkeypatch.setenv("MOBILERUN_CONFIG", str(tmp_path))
    else:
        args.extend(["-c", str(tmp_path)])
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 2, result.output
    assert "is a directory" in result.stderr
    assert "'-c' / '--config'" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("content", ["device:\n  platform: android\n", ""])
def test_device_list_refuses_config_without_version(monkeypatch, tmp_path, content):
    config = tmp_path / "old.yaml"
    config.write_text(content)
    monkeypatch.setenv("MOBILERUN_CONFIG", str(config))
    client = Mock()
    monkeypatch.setattr(device_lifecycle, "core_client", client)
    result = CliRunner().invoke(cli, ["device", "list"])
    assert result.exit_code == 1, result.output
    assert "outdated (missing _version)" in result.stderr
    client.assert_not_called()


def test_delegate_precondition_precedes_banner():
    result = CliRunner().invoke(cli, ["run", "x", "--delegate-model", "openai/x"])
    assert result.exit_code == 2, result.output
    assert "--delegate-model needs a decision model" in result.stderr
    assert "Try Mobilerun Cloud" not in result.output
    assert "Starting:" not in result.output
    assert result.stdout == ""


def test_delegate_precondition_uses_overridden_reasoning_role(monkeypatch):
    from mobilerun.config_manager.config_manager import DecisionProfile

    config = MobileConfig()
    config.decision_profiles["fast_agent"] = DecisionProfile()
    monkeypatch.setattr(ConfigLoader, "load", lambda _: config)
    result = CliRunner().invoke(cli, ["run", "x", "-r", "--delegate-model", "openai/x"])
    assert result.exit_code == 2, result.output
    assert "--delegate-model needs a decision model" in result.stderr
    assert "Starting:" not in result.output


@pytest.fixture
def rate_limit_server():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class FakePortalAndLLM(BaseHTTPRequestHandler):
        llm_requests = []

        def log_message(self, *args):
            pass

        def reply(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if status == 429:
                self.send_header("Retry-After", "2")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/state_full"):
                self.reply(dict(a11y_tree={}, phone_state={}, device_context={}))
            else:
                self.reply({"version": "0.7.25"})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path.startswith("/v1/"):
                self.llm_requests.append(self.path)
                self.reply(
                    {
                        "error": {
                            "message": "persistent rate limit (429)",
                            "type": "rate_limit_error",
                        }
                    },
                    429,
                )
            else:
                self.reply({"result": 1730000000000 if self.path == "/time" else {}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakePortalAndLLM)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", FakePortalAndLLM.llm_requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "provider,model",
    [("openai-like", "gpt-6.1-sol"), ("anthropic", "claude-sonnet-5-5")],
)
def test_run_persistent_429_is_bounded_and_emits_one_document(
    tmp_path, rate_limit_server, provider, model
):
    import os
    import subprocess
    import sys

    url, requests = rate_limit_server
    path = ConfigLoader.save(MobileConfig(), tmp_path / "run.yaml")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("MOBILERUN_", "DROIDRUN_", "OPENAI_", "ANTHROPIC_"))
        and not key.endswith(("_API_KEY", "_TOKEN"))
    }
    env.update(
        HOME=str(tmp_path),
        MOBILERUN_TELEMETRY_ENABLED="false",
        OPENAI_API_KEY="fake-local-key",
        ANTHROPIC_API_KEY="fake-local-key",
        MOBILERUN_ANDROID_PORTAL_TOKEN="fake-local-token",
        HTTP_PROXY="http://127.0.0.1:1",
        HTTPS_PROXY="http://127.0.0.1:1",
        ALL_PROXY="http://127.0.0.1:1",
        NO_PROXY="127.0.0.1,localhost",
    )
    args = ["run", "x", "-c", str(path), "-d", url]
    args += "-P android -s 3 -j".split()
    args += [
        "-p",
        provider,
        "-m",
        model,
        "-u",
        url + ("/v1" if provider == "openai-like" else ""),
    ]
    result = subprocess.run(
        [sys.executable, "-c", "from mobilerun.cli.main import cli; cli()", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "429" in result.stderr
    assert (
        "/v1/messages" if provider == "anthropic" else "/v1/chat/completions"
    ) in requests
    document = json.loads(result.stdout)
    assert document["success"] is False
    assert "429" in document["reason"]


def test_retry_defaults_cover_provider_families_and_legacy_adapters():
    from mobilerun.cli.providers import llm_retry_kwargs

    runtimes = (
        "OpenAI OpenAIResponses OpenAILike Anthropic GoogleGenAI XAI ZAI ZAI_Coding "
        "MiniMax DeepSeek OpenRouter openai_oauth xai_oauth"
    )
    for runtime in runtimes.split():
        assert llm_retry_kwargs(runtime) == {"max_retries": 1}, runtime
    for runtime in ("Ollama", "anthropic_oauth", "gemini_oauth_code_assist"):
        assert llm_retry_kwargs(runtime) == {}, runtime


@pytest.mark.parametrize("error", [TimeoutError(), AdbTimeout("adb read timeout")])
def test_adb_timeouts_use_the_shared_hint(error):
    import click

    with (
        pytest.raises(click.ClickException, match="Platform-Tools"),
        selection.local_adb_errors(),
    ):
        raise error


def test_device_list_bounds_an_unresponsive_adb_probe(monkeypatch):
    import asyncio

    cancelled = Mock()

    async def stalled():
        try:
            await asyncio.sleep(10)
            return []
        finally:
            cancelled()

    client = SimpleNamespace(list_devices=Mock(return_value=[]), close=Mock())
    monkeypatch.setattr(device_lifecycle, "core_client", lambda *a, **kw: client)
    monkeypatch.setattr(device_lifecycle.adb, "list", stalled)
    result = CliRunner().invoke(cli, ["device", "list", "-j"])
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout) == {"error": ADB_HINT}
    assert result.stderr == "Error: " + ADB_HINT + "\n"
    cancelled.assert_called_once()
    client.close.assert_called_once()
