"""Regression coverage for CLI safety, transport selection and lightweight help."""

import asyncio
import importlib
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.cli.devices.base import options, selection
from mobilerun.config.user_config import MobileConfig
from mobilerun.core.devices.base import Device
from mobilerun.core.extensions import Extension, action

SECRET = "SECRET-MARKER-DO-NOT-PRINT"
CLOUD = "12345678-1234-1234-1234-123456789abc"


class FailingExtension(Extension):
    @action()
    async def fail(self) -> None:
        """Fail unexpectedly."""
        raise RuntimeError(SECRET)


def test_actual_action_failure_is_sanitized(monkeypatch):
    phone = Device()
    asyncio.run(FailingExtension(name="demo").bind(phone))
    phone.close = AsyncMock()
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=phone))
    args = ["device", "action", "call", "demo.fail", "serial"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1 and SECRET not in result.output
    assert "RuntimeError" in result.output
    phone.close.assert_awaited_once()


def test_call_connection_failure_reaches_root_sanitizer(monkeypatch):
    monkeypatch.setattr(
        selection, "connect_target", AsyncMock(side_effect=RuntimeError(SECRET))
    )
    args = ["device", "action", "call", "demo.fail", "serial"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1 and SECRET not in result.output
    assert "--debug" in result.output
    debug = CliRunner().invoke(cli, ["--debug", *args])
    assert debug.exit_code == 1 and SECRET in debug.output


@pytest.mark.parametrize(
    "module_name, boundary, args",
    [
        ("auth.login", "deviceauth.request_code", ["account", "login"]),
        ("auth.status", "deviceauth.fetch_session", ["account", "status"]),
        ("devices.connect", "adb.connect", ["device", "connect", "host:5555"]),
        ("devices.disconnect", "adb.disconnect", ["device", "disconnect", "host:5555"]),
        ("devices.list", "discover_devices", ["device", "list"]),
        ("devices.ping", "adb.device", ["device", "ping", "serial"]),
        ("devices.setup", "adb.device", ["device", "setup", "serial"]),
    ],
)
def test_unexpected_boundary_failures_do_not_print_values(
    monkeypatch, module_name, boundary, args
):
    module = importlib.import_module("mobilerun.cli." + module_name)
    parent, _, attribute = boundary.rpartition(".")
    owner = getattr(module, parent) if parent else module
    monkeypatch.setattr(owner, attribute, AsyncMock(side_effect=RuntimeError(SECRET)))
    if module_name == "auth.status":
        monkeypatch.setattr(
            module.credentials, "load_credentials", lambda: {"access_token": "fake"}
        )
    if module_name == "devices.connect":
        monkeypatch.setattr(module, "discover_devices", AsyncMock(return_value=[]))
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1, result.output
    assert SECRET not in result.output and "--debug" in result.output


@pytest.mark.parametrize(
    "platform, backend, expected",
    [
        ("ios", None, "ios"),
        ("android", None, "android"),
        ("ios", "visual-remote", "visual_remote"),
    ],
)
def test_saved_url_uses_configured_transport_without_discovery(
    monkeypatch, platform, backend, expected
):
    config = MobileConfig()
    config.device.serial = "http://localhost:8080"
    config.device.platform = platform
    config.device.control_backend = backend
    monkeypatch.setattr(options.ConfigLoader, "load", lambda path: config)
    discover = AsyncMock(
        side_effect=AssertionError("URL endpoints aren't discoverable")
    )
    monkeypatch.setattr(options, "discover_devices", discover)
    phone = AsyncMock()
    connect = AsyncMock(return_value=phone)
    monkeypatch.setattr(selection, "connect_target", connect)
    result = CliRunner().invoke(cli, ["device", "input", "press", "HOME"])
    assert result.exit_code == 0, result.output
    assert connect.await_args.args == (config.device.serial,)
    assert connect.await_args.kwargs["kind"] == expected
    phone.close.assert_awaited_once()
    discover.assert_not_awaited()


def test_agent_run_ios_without_serial_discovers_portal(monkeypatch):
    pytest.importorskip("mobilerun_agents", reason="requires mobilerun-agents add-on")
    from mobilerun_core_local.driver import ios

    module = importlib.import_module("mobilerun_agents.cli.agent.run")
    config = MobileConfig()
    monkeypatch.setattr(options.ConfigLoader, "load", lambda path: config)
    discovered = AsyncMock(return_value="http://localhost:8080")
    monkeypatch.setattr(ios, "discover_ios_device", discovered)
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(side_effect=AssertionError("No ADB for iOS")),
    )
    connect = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(selection, "connect_target", connect)
    execute = AsyncMock(return_value=NS(answer="done", success=True))
    monkeypatch.setattr(module, "AgentOperator", lambda **kwargs: NS(execute=execute))
    monkeypatch.setattr(module.ConfigLoader, "load", lambda _: NS())
    result = CliRunner().invoke(cli, ["agent", "run", "hello", "--ios"])
    assert result.exit_code == 0, result.output
    discovered.assert_awaited_once_with()
    assert connect.await_args.kwargs["kind"] == "ios"
    execute.assert_awaited_once_with(connect.return_value, "hello", max_steps=None)


@pytest.mark.parametrize("operation", ["resume", "reboot"])
def test_chargeable_lifecycle_requires_consent_before_client(monkeypatch, operation):
    module = importlib.import_module("mobilerun.cli.devices.cloud." + operation)
    calls = []

    @asynccontextmanager
    async def client(**kwargs):
        calls.append(kwargs)
        yield object()

    method = AsyncMock()
    monkeypatch.setattr(module, "cloud_client", client)
    monkeypatch.setattr(module.devices, operation, method)
    result = CliRunner().invoke(cli, ["device", "cloud", operation, CLOUD])
    assert result.exit_code == 1 and "--yes" in result.output
    assert calls == []
    method.assert_not_awaited()
    accepted = CliRunner().invoke(cli, ["device", "cloud", operation, CLOUD, "--yes"])
    assert accepted.exit_code == 0, accepted.output
    method.assert_awaited_once()


@pytest.mark.parametrize("operation", ["resume", "reboot"])
def test_lifecycle_warning_and_decline_prevent_billing(monkeypatch, operation):
    from mobilerun.cli.base import menu

    module = importlib.import_module("mobilerun.cli.devices.cloud." + operation)
    client = AsyncMock(side_effect=AssertionError("No client before confirmation"))
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    from mobilerun.cli.base import safety

    monkeypatch.setattr(safety, "confirm_prompt_async", AsyncMock(return_value=False))
    monkeypatch.setattr(module, "cloud_client", client)
    result = CliRunner().invoke(cli, ["device", "cloud", operation, CLOUD], input="n\n")
    assert result.exit_code == 0, result.output
    assert "may incur charges" in result.output and "Billing: paid" in result.output
    client.assert_not_called()


def test_stop_remains_nonbillable_and_does_not_require_yes(monkeypatch):
    module = importlib.import_module("mobilerun.cli.devices.cloud.stop")

    @asynccontextmanager
    async def client(**kwargs):
        yield object()

    method = AsyncMock()
    monkeypatch.setattr(module, "cloud_client", client)
    monkeypatch.setattr(module.devices, "stop", method)
    result = CliRunner().invoke(cli, ["device", "cloud", "stop", CLOUD])
    assert result.exit_code == 0, result.output
    method.assert_awaited_once()


@pytest.mark.parametrize(
    "args",
    [
        ["agent", "--help"],
        ["agent", "task", "--help"],
        ["agent", "task", "status", "--help"],
    ],
)
def test_agent_management_help_follows_plugin_availability(args):
    script = f"""
import sys
from click.testing import CliRunner
from mobilerun.addons.plugins import discover
from mobilerun.addons.settings import CLI_ENTRY_POINT_GROUP
from mobilerun.cli.app import cli
r = CliRunner().invoke(cli, {args!r})
if ("agent" in discover(CLI_ENTRY_POINT_GROUP)):
    assert r.exit_code == 0, r.output
    assert 'Commands' in r.output or 'Options' in r.output
else:
    assert r.exit_code != 0
    assert 'Install the mobilerun-agents add-on (see README: Agent add-on).' in r.output
    assert not any(m.startswith(('mobilerun.agents', 'mobilerun_agents')) for m in sys.modules)
    assert not any(m == 'llama_index' or m.startswith('llama_index.') for m in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["countries", "messages"])
def test_number_read_views_use_visual_language_and_keep_json(monkeypatch, kind):
    module = importlib.import_module("mobilerun.cli.phone_number.group")
    entry = (
        {"country": "US", "name": "United States", "in_stock": True, "plan_id": "plan"}
        if kind == "countries"
        else {
            "id": "sms-1",
            "peer_number": "+123",
            "body": "Hello",
            "status": "sent",
            "direction": "outbound",
            "occurred_at": "2026-11-20T10:00:00Z",
        }
    )
    model = NS(model_dump=lambda **kwargs: entry)
    value = NS(data=NS(items=[model])) if kind == "countries" else [model]
    monkeypatch.setattr(module.cloud_numbers, kind, AsyncMock(return_value=value))
    args = ["number", kind] + (["number-1"] if kind == "messages" else [])
    human = CliRunner().invoke(cli, args)
    assert human.exit_code == 0, human.output
    assert ("Countries · 1" if kind == "countries" else "Messages · 1") in human.output
    assert "{" not in human.output
    machine = CliRunner().invoke(cli, [*args, "--json"])
    assert json.loads(machine.stdout) == [entry]


def test_workspace_upload_acknowledgements(monkeypatch, tmp_path):
    module = importlib.import_module("mobilerun.cli.workspace.group")
    path = tmp_path / "app.apk"
    path.write_bytes(b"fake")
    monkeypatch.setattr(
        module.Workspace,
        "upload_file",
        AsyncMock(return_value=NS(id="file-1", name="app.apk")),
    )
    monkeypatch.setattr(
        module.Workspace,
        "upload_app",
        AsyncMock(return_value=NS(id="app-1", name="Example", package="com.example")),
    )
    for group, label in [("files", "app.apk"), ("apps", "Example")]:
        args = ["workspace", group, "upload", str(path)]
        human = CliRunner().invoke(cli, args)
        assert human.exit_code == 0, human.output
        assert f"✓ Uploaded {label}" in human.stdout
        machine = CliRunner().invoke(cli, [*args, "--json"])
        data = json.loads(machine.stdout)
        assert (data[0] if isinstance(data, list) else data)["status"] == "uploaded"
