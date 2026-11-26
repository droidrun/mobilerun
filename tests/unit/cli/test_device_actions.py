"""Device CLI actions, extension dispatch, cloud safety, and transport selection."""

import asyncio
import importlib
import json
import subprocess
import sys
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner
from mobilerun_sdk.types import DeviceRetrieveResponse

from mobilerun.cli.app import cli
from mobilerun.cli.devices.base import discovery, options, selection
from mobilerun.cli.devices.group import device
from mobilerun.clients.cloud import devices
from mobilerun.config.settings import get_settings
from mobilerun.config.user_config import MobileConfig
from mobilerun.core.devices.base import Device
from mobilerun.core.extensions import Extension, action
from mobilerun.errors import (
    BillingRequired,
    CloudPermissionDenied,
    DeviceOperationNotSupported,
)
from mobilerun.models.billing import Billing
from mobilerun.models.context import ContextItem


@pytest.mark.parametrize(
    ("args", "method", "values"),
    [
        (
            ["settings", "location", "set", "52.52", "13.40"],
            "set_location",
            (52.52, 13.4),
        ),
        (["settings", "location", "reset"], "reset_location", ()),
        (
            ["settings", "timezone", "set", "Europe/Berlin"],
            "set_timezone",
            ("Europe/Berlin",),
        ),
        (["app", "open", "https://example.com"], "open_url", ("https://example.com",)),
    ],
)
def test_environment_cli_selects_device(monkeypatch, args, method, values):
    from mobilerun.cli.devices.base import selection as environment

    phone = SimpleNamespace(close=AsyncMock())
    setattr(
        phone,
        method,
        AsyncMock(return_value="clip" if method == "get_clipboard" else None),
    )
    connect = AsyncMock(return_value=phone)
    monkeypatch.setattr(environment, "connect_target", connect)
    result = CliRunner().invoke(device, [*args, "--device", "emulator-5554"])
    assert result.exit_code == 0, (result.output, result.exception)
    getattr(phone, method).assert_awaited_once_with(*values)
    connect.assert_awaited_once()
    assert connect.await_args is not None
    assert connect.await_args.args[0] == "emulator-5554"
    phone.close.assert_awaited_once()
    if method == "get_clipboard":
        assert "clip" in result.output


def test_unsupported_is_a_click_error(monkeypatch):
    from mobilerun.cli.devices.base import selection as environment

    phone = SimpleNamespace(
        close=AsyncMock(),
        reset_location=AsyncMock(
            side_effect=DeviceOperationNotSupported("no reset\ntry again")
        ),
    )
    monkeypatch.setattr(environment, "connect_target", AsyncMock(return_value=phone))
    result = CliRunner().invoke(
        device, ["settings", "location", "reset", "--device", "emulator-5554"]
    )
    assert result.exit_code == 1
    assert result.output.strip() == "Error: no reset try again"
    phone.close.assert_awaited_once()


@pytest.mark.parametrize("args", [["get"], ["set", "text"]])
@pytest.mark.parametrize(
    "target_args", [["emulator-5554"], ["http://ios.test", "--ios"]]
)
def test_clipboard_rejects_local_before_connect(
    monkeypatch: pytest.MonkeyPatch, args: list[str], target_args: list[str]
) -> None:
    phone = Device()
    get_clipboard = AsyncMock(wraps=phone.get_clipboard)
    set_clipboard = AsyncMock(wraps=phone.set_clipboard)
    monkeypatch.setattr(phone, "get_clipboard", get_clipboard)
    monkeypatch.setattr(phone, "set_clipboard", set_clipboard)
    connect = AsyncMock(return_value=phone)
    monkeypatch.setattr(selection, "connect_target", connect)

    result = CliRunner().invoke(
        device, ["input", "clipboard", *args, "--device", *target_args]
    )

    assert result.exit_code == 2, result.output
    assert "Clipboard commands require a cloud device ID." in result.stderr
    assert "Traceback" not in result.output
    connect.assert_not_called()
    get_clipboard.assert_not_called()
    set_clipboard.assert_not_called()


def test_set_language_on_local_reports_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect = AsyncMock(return_value=Device())
    monkeypatch.setattr(selection, "connect_target", connect)
    result = CliRunner().invoke(
        device, ["settings", "language", "set", "de-DE", "--device", "emulator-5554"]
    )
    assert result.exit_code == 1
    assert result.output.strip() == "Error: Device doesn't support set_language"
    connect.assert_awaited_once()


@pytest.mark.parametrize(
    ("args", "method", "params", "output"),
    [
        (["settings", "language", "set", "de-DE"], "language", {"locale": "de-DE"}, ""),
        (
            ["input", "clipboard", "set", "hello"],
            "clipboard_set",
            {"text": "hello"},
            "",
        ),
        (["input", "clipboard", "get"], "clipboard_get", {}, "hello"),
    ],
)
def test_cloud_only_environment_builds_typed_selected_phone(
    monkeypatch, args, method, params, output
):
    from mobilerun.core.devices import cloud as environment
    from mobilerun.core.devices.cloud import CloudDevice
    from mobilerun.core.proxy import ProxyStatus

    target = "11111111-1111-1111-1111-111111111111"
    phone = CloudDevice(target, api_key="test")
    monkeypatch.setattr(
        phone, "_proxy_status", AsyncMock(return_value=ProxyStatus(False))
    )
    calls = []
    endpoint = AsyncMock(return_value=SimpleNamespace(text="hello"))
    resources = SimpleNamespace(
        retrieve_capabilities=AsyncMock(
            return_value=SimpleNamespace(
                capabilities=SimpleNamespace(telephony=False, proxy=True)
            )
        ),
        number=SimpleNamespace(
            retrieve=AsyncMock(return_value=SimpleNamespace(binding=None))
        ),
        language=SimpleNamespace(set=endpoint),
        clipboard=SimpleNamespace(set=endpoint, get=endpoint),
    )
    monkeypatch.setattr(phone, "_client", SimpleNamespace(devices=resources))
    monkeypatch.setattr(phone, "_write_client", phone._client)
    close = AsyncMock()
    monkeypatch.setattr(phone, "close", close)

    def build(*, device_id, base_url):
        calls.append((device_id, base_url))
        return phone

    monkeypatch.setattr(environment, "CloudDevice", build)
    result = CliRunner().invoke(
        device, [*args, "--device", target, "--base-url", "https://example.invalid"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == [(target, "https://example.invalid")]
    endpoint.assert_awaited_once_with(target, **params)
    close.assert_awaited_once()
    if output:
        assert output in result.output
    if method == "clipboard_set":
        assert f"Clipboard set on {target}" in result.stdout
        assert params["text"] not in result.output


def test_location_prompts_for_device_before_coordinates(monkeypatch):
    from mobilerun.cli.base import menu
    from mobilerun.cli.devices.base import options
    from mobilerun.cli.devices.base import selection as environment
    from mobilerun.cli.devices.base.discovery import DiscoveredDevice

    order = []
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(
            return_value=[
                DiscoveredDevice("emulator-5554", "local", None, "device", None),
                DiscoveredDevice("emulator-5556", "local", None, "device", None),
            ]
        ),
    )
    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    def pick(*args):
        order.append("device")
        return "emulator-5554"

    def text(label, **kwargs):
        order.append(label)
        return "1"

    monkeypatch.setattr(options, "select_prompt", pick)
    monkeypatch.setattr(menu, "text_prompt", text)
    monkeypatch.setattr(options, "confirm_prompt", lambda *args, **kwargs: False)
    phone = SimpleNamespace(close=AsyncMock(), set_location=AsyncMock())
    monkeypatch.setattr(environment, "connect_target", AsyncMock(return_value=phone))
    result = CliRunner().invoke(device, ["settings", "location", "set"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert order == ["device", "Latitude", "Longitude"]
    phone.set_location.assert_awaited_once_with(1.0, 1.0)


def test_unsupported_connect_is_a_click_error(monkeypatch):
    from mobilerun.cli.devices.base import selection as environment

    monkeypatch.setattr(
        environment,
        "connect_target",
        AsyncMock(side_effect=DeviceOperationNotSupported("cannot connect")),
    )
    result = CliRunner().invoke(
        device, ["app", "open", "https://example.com", "--device", "emulator-5554"]
    )
    assert result.exit_code == 1
    assert result.output.strip() == "Error: cannot connect"


class DemoExtension(Extension):
    default_name = "demo"
    exclusive = "example"

    async def context(self):
        return [ContextItem("demo", "info", {"message": "ready"})]

    @action()
    async def repeat(self, text: str, count: int, loud: bool = False) -> str:
        """Repeat a message."""
        result = text * count
        return result.upper() if loud else result


@pytest.fixture
def connected(monkeypatch):
    from mobilerun.cli.devices.base import selection as extensions

    device = Device()
    asyncio.run(DemoExtension().bind(device))
    device.close = AsyncMock()
    monkeypatch.setattr(extensions, "connect_target", AsyncMock(return_value=device))
    return device


def test_extensions_json_reports_context(connected):
    result = CliRunner().invoke(
        device, ["action", "extensions", "emulator-5554", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [
        {
            "name": "demo",
            "kind": "DemoExtension",
            "exclusive": "example",
            "context": [
                {"source": "demo", "kind": "info", "data": {"message": "ready"}}
            ],
        }
    ]
    connected.close.assert_awaited_once()


def test_actions_lists_public_actions(connected):
    result = CliRunner().invoke(device, ["action", "list", "emulator-5554"])
    assert result.exit_code == 0, result.output
    assert "demo.repeat" in result.output and "Repeat a message" in result.output
    connected.close.assert_awaited_once()


def test_call_converts_types_and_serializes(connected):
    result = CliRunner().invoke(
        device,
        [
            "action",
            "call",
            "demo.repeat",
            "emulator-5554",
            "--arg",
            "text=ab",
            "--arg",
            "count=2",
            "--arg",
            "loud=true",
            "--json",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(result.output) == "ABAB"
    connected.close.assert_awaited_once()


def test_call_missing_arg_is_clean_error(connected):
    result = CliRunner().invoke(
        device, ["action", "call", "demo.repeat", "emulator-5554", "--arg", "text=x"]
    )
    assert result.exit_code == 1
    assert "count" in result.output
    connected.close.assert_awaited_once()


def test_call_prompts_required_argument(connected, monkeypatch):
    import importlib

    extensions = importlib.import_module("mobilerun.cli.devices.action.call")

    monkeypatch.setattr(extensions, "is_interactive", lambda: True)
    monkeypatch.setattr(extensions, "text_prompt", AsyncMock(return_value="2"))
    result = CliRunner().invoke(
        device,
        ["action", "call", "demo.repeat", "emulator-5554", "--arg", "text=x", "--json"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(result.output) == "xx"


def test_call_rejects_unknown_action_and_closes(connected):
    result = CliRunner().invoke(
        device, ["action", "call", "demo.missing", "emulator-5554"]
    )
    assert result.exit_code == 1
    assert result.output.strip() == "Error: Unknown action: demo.missing"
    connected.close.assert_awaited_once()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--arg", "oops"],
        ["--arg", "unknown=x"],
        ["--arg", "count=bad", "--arg", "text=x"],
    ],
)
def test_call_bad_arguments_are_clean_errors(connected, arguments):
    result = CliRunner().invoke(
        device, ["action", "call", "demo.repeat", "emulator-5554", *arguments]
    )
    assert result.exit_code == 1
    assert result.output.startswith("Error: ")
    connected.close.assert_awaited_once()


def test_connection_error_is_one_line(monkeypatch):
    from mobilerun.cli.devices.base import selection as extensions

    monkeypatch.setattr(
        extensions,
        "connect_target",
        AsyncMock(side_effect=RuntimeError("offline\ntry later")),
    )
    result = CliRunner().invoke(device, ["action", "list", "emulator-5554"])
    assert result.exit_code == 1
    assert "--debug" in result.output


CLOUD = "11111111-1111-1111-1111-111111111111"


def test_status_json_only_prints_status_fields(monkeypatch):
    module = importlib.import_module("mobilerun.cli.devices.cloud.status")
    info = DeviceRetrieveResponse.model_construct(
        id=CLOUD,
        name="Phone",
        state="ready",
        type="android_cloud_phone",
        streamToken="synthetic-token",
        streamUrl="https://example.invalid/synthetic-token",
        futureSecret="synthetic-token",
    )
    monkeypatch.setattr(module, "cloud_client", lambda **kw: nullcontext(object()))
    monkeypatch.setattr(devices, "retrieve", AsyncMock(return_value=info))
    result = CliRunner().invoke(device, ["cloud", "status", CLOUD, "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["state"] == "ready"
    assert "streamToken" not in result.output and "stream_token" not in result.output
    assert "synthetic-token" not in result.output


@pytest.mark.parametrize(
    "mode,flags,expected",
    [
        ("no_commerce", [], "paid"),
        ("included", ["--yes"], "paid"),
        ("paid", [], "--yes"),
        ("paid", ["--yes"], "sent"),
    ],
)
@pytest.mark.parametrize("alias", ["sms", "work"])
def test_third_party_sms_retains_legacy_confirmation(
    monkeypatch, mode, flags, expected, alias
):
    sent = []

    class BoundNumber(Extension):
        role = "phone_number"

        @action()
        async def send(self, to: str, text: str) -> str:
            """Record an outgoing SMS without calling a provider."""
            sent.append((to, text))
            return "sent"

    phone = Device()
    asyncio.run(BoundNumber(name=alias).bind(phone))
    phone.close = AsyncMock()
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=phone))
    monkeypatch.setenv("MOBILERUN_BILLING", mode)
    args = [
        "action",
        "call",
        f"{alias}.send",
        CLOUD,
        "--arg",
        "to=5550100",
        "--arg",
        "text=hello",
        *flags,
    ]
    result = CliRunner().invoke(device, args)
    assert expected in result.output
    assert (result.exit_code == 0) == (expected == "sent")
    assert sent == ([("5550100", "hello")] if expected == "sent" else [])


@pytest.mark.parametrize("mode", [Billing.INCLUDED, Billing.NO_COMMERCE])
@pytest.mark.parametrize("strategy", ["minute", None, "auto", "subscription"])
def test_restricted_resume_checks_existing_strategy(mode, strategy):
    client = SimpleNamespace(
        devices=SimpleNamespace(
            retrieve=AsyncMock(return_value=SimpleNamespace(billing_strategy=strategy)),
            resume=AsyncMock(),
        )
    )
    if strategy == "subscription":
        asyncio.run(devices.resume(CLOUD, billing=mode, client=client))
        client.devices.resume.assert_awaited_once_with(CLOUD)
    else:
        with pytest.raises(BillingRequired, match="paid") as caught:
            asyncio.run(devices.resume(CLOUD, billing=mode, client=client))
        assert (caught.value.needed, caught.value.given) == (Billing.PAID, mode)
        client.devices.resume.assert_not_awaited()
    client.devices.retrieve.assert_awaited_once_with(CLOUD)


def test_restricted_subscription_resume_never_prompts(monkeypatch):
    from mobilerun.cli.base import menu, safety

    monkeypatch.setenv("MOBILERUN_BILLING", "included")
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    prompt = AsyncMock(side_effect=AssertionError("subscription needs no consent"))
    monkeypatch.setattr(safety, "confirm_prompt_async", prompt)
    transport = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=SimpleNamespace(billing_strategy="subscription")
        ),
        resume=AsyncMock(),
    )
    module = importlib.import_module("mobilerun.cli.devices.cloud.resume")
    monkeypatch.setattr(
        module,
        "cloud_client",
        lambda **kwargs: nullcontext(SimpleNamespace(devices=transport)),
    )
    result = CliRunner().invoke(device, ["cloud", "resume", CLOUD])
    assert result.exit_code == 0, result.output
    prompt.assert_not_awaited()
    transport.retrieve.assert_awaited_once_with(CLOUD)
    transport.resume.assert_awaited_once_with(CLOUD)


def test_paid_resume_needs_no_strategy_lookup():
    client = SimpleNamespace(devices=SimpleNamespace(resume=AsyncMock()))
    asyncio.run(devices.resume(CLOUD, billing=Billing.PAID, client=client))
    client.devices.resume.assert_awaited_once_with(CLOUD)


def test_strict_discovery_withholds_unexpected_details_except_in_debug(monkeypatch):
    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: "fake")
    monkeypatch.setattr(
        discovery,
        "list_devices",
        AsyncMock(side_effect=RuntimeError("diagnostic-canary")),
    )
    args = ["device", "list", "--cloud"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1 and "Error listing cloud devices" in result.stderr
    assert "diagnostic-canary" not in result.stderr
    result = CliRunner().invoke(cli, ["--debug", *args])
    assert result.exit_code == 1 and "diagnostic-canary" in result.stderr


@pytest.fixture
def rename_setup(monkeypatch):
    module = importlib.import_module("mobilerun.cli.devices.cloud.rename")
    endpoint = AsyncMock()
    monkeypatch.setattr(module.devices, "set_name", endpoint)
    opened = object()
    calls = []

    @asynccontextmanager
    async def client(**kwargs):
        calls.append(kwargs)
        yield opened

    monkeypatch.setattr(module, "cloud_client", client)
    return module, endpoint, opened, calls


def test_rename_arguments_and_json(rename_setup):
    _, endpoint, opened, calls = rename_setup
    result = CliRunner().invoke(
        device,
        [
            "cloud",
            "rename",
            CLOUD,
            "  New name  ",
            "--json",
            "--base-url",
            "https://example.test",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(result.stdout) == {"id": CLOUD, "name": "New name"}
    billing = get_settings().MOBILERUN_BILLING
    assert calls == [{"base_url": "https://example.test", "billing": billing}]
    endpoint.assert_awaited_once_with(CLOUD, "New name", billing=billing, client=opened)


def test_rename_interactive_prompt(rename_setup, monkeypatch):
    module, endpoint, _, _ = rename_setup
    monkeypatch.setattr(module, "is_interactive", lambda: True)
    monkeypatch.setattr(module, "text_prompt_async", AsyncMock(return_value=" Phone "))
    result = CliRunner().invoke(device, ["cloud", "rename", CLOUD])
    assert result.exit_code == 0, (result.output, result.exception)
    assert f"✓ Renamed {CLOUD} to Phone" in result.output
    assert endpoint.await_args.args == (CLOUD, "Phone")


def test_rename_empty_prompt_cancels(rename_setup, monkeypatch):
    module, endpoint, _, _ = rename_setup
    monkeypatch.setattr(module, "is_interactive", lambda: True)
    monkeypatch.setattr(module, "text_prompt_async", AsyncMock(return_value="  "))
    result = CliRunner().invoke(device, ["cloud", "rename", CLOUD])
    assert result.exit_code == 0
    assert result.stderr == "Cancelled.\n"
    endpoint.assert_not_awaited()


@pytest.mark.parametrize("args", [[CLOUD, "  "], [CLOUD]])
def test_rename_invalid_noninteractive_name(rename_setup, args):
    _, endpoint, _, _ = rename_setup
    result = CliRunner().invoke(device, ["cloud", "rename", *args])
    assert result.exit_code != 0
    endpoint.assert_not_awaited()


def test_rename_picks_cloud_device_when_omitted(rename_setup, monkeypatch):
    from mobilerun.cli.base import choices
    from mobilerun.cli.base.prompts import SelectChoice
    from mobilerun.cli.devices.base import options, pickers

    module, endpoint, _, _ = rename_setup
    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(pickers, "is_interactive", lambda: True)
    monkeypatch.setattr(module, "is_interactive", lambda: True)
    monkeypatch.setattr(module, "text_prompt_async", AsyncMock(return_value="New"))
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(side_effect=AssertionError("local discovery")),
    )
    monkeypatch.setattr(
        choices, "cloud_devices", lambda ctx: [SelectChoice(CLOUD, "Phone")]
    )
    monkeypatch.setattr(
        pickers, "select_prompt", lambda label, entries: entries[0].value
    )
    result = CliRunner().invoke(device, ["cloud", "rename"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert endpoint.await_args.args == (CLOUD, "New")


@pytest.mark.parametrize("json_option", [[], ["--json"]])
def test_rename_cloud_error_is_reported_on_stderr(rename_setup, json_option):
    _, endpoint, _, _ = rename_setup
    error = CloudPermissionDenied(operation="Rename cloud phone", status=403)
    endpoint.side_effect = error
    result = CliRunner().invoke(device, ["cloud", "rename", CLOUD, "New", *json_option])
    assert result.exit_code != 0
    assert str(error) in result.stderr
    assert result.stdout == ""


def test_set_name_preserves_injected_client_and_sanitizes_error():
    import asyncio
    from types import SimpleNamespace

    import httpx
    from mobilerun_sdk import APIStatusError

    from mobilerun.clients.cloud import devices
    from mobilerun.errors import CloudPermissionDenied

    error = APIStatusError(
        "sensitive-value",
        response=httpx.Response(
            403, request=httpx.Request("PUT", "https://example.test")
        ),
        body={},
    )
    endpoint = AsyncMock(side_effect=error)
    client = SimpleNamespace(
        devices=SimpleNamespace(set_name=endpoint), close=AsyncMock()
    )
    with pytest.raises(CloudPermissionDenied) as caught:
        asyncio.run(
            devices.set_name(CLOUD, "name", billing="no_commerce", client=client)
        )
    assert "sensitive-value" not in str(caught.value)
    client.close.assert_not_awaited()


SECRET = "SECRET-MARKER-DO-NOT-PRINT"


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
    execute = AsyncMock(return_value=SimpleNamespace(answer="done", success=True))
    monkeypatch.setattr(
        module, "AgentOperator", lambda **kwargs: SimpleNamespace(execute=execute)
    )
    monkeypatch.setattr(module.ConfigLoader, "load", lambda _: SimpleNamespace())
    result = CliRunner().invoke(cli, ["agent", "run", "hello", "--ios"])
    assert result.exit_code == 0, result.output
    discovered.assert_awaited_once_with()
    assert connect.await_args.kwargs["kind"] == "ios"
    execute.assert_awaited_once_with(connect.return_value, "hello", max_steps=None)


def test_reboot_requires_consent_before_client(monkeypatch):
    module = importlib.import_module("mobilerun.cli.devices.cloud.reboot")
    calls = []

    @asynccontextmanager
    async def client(**kwargs):
        calls.append(kwargs)
        yield object()

    method = AsyncMock()
    monkeypatch.setattr(module, "cloud_client", client)
    monkeypatch.setattr(module.devices, "reboot", method)
    result = CliRunner().invoke(cli, ["device", "cloud", "reboot", CLOUD])
    assert result.exit_code == 1 and "--yes" in result.output
    assert calls == []
    method.assert_not_awaited()
    accepted = CliRunner().invoke(cli, ["device", "cloud", "reboot", CLOUD, "--yes"])
    assert accepted.exit_code == 0, accepted.output
    method.assert_awaited_once()


def test_reboot_warning_and_decline_prevent_billing(monkeypatch):
    from mobilerun.cli.base import menu

    module = importlib.import_module("mobilerun.cli.devices.cloud.reboot")
    client = AsyncMock(side_effect=AssertionError("No client before confirmation"))
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    from mobilerun.cli.base import safety

    monkeypatch.setattr(safety, "confirm_prompt_async", AsyncMock(return_value=False))
    monkeypatch.setattr(module, "cloud_client", client)
    result = CliRunner().invoke(cli, ["device", "cloud", "reboot", CLOUD], input="n\n")
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
    model = SimpleNamespace(model_dump=lambda **kwargs: entry)
    value = (
        SimpleNamespace(data=SimpleNamespace(items=[model]))
        if kind == "countries"
        else [model]
    )
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
        AsyncMock(return_value=SimpleNamespace(id="file-1", name="app.apk")),
    )
    monkeypatch.setattr(
        module.Workspace,
        "upload_app",
        AsyncMock(
            return_value=SimpleNamespace(
                id="app-1", name="Example", package="com.example"
            )
        ),
    )
    for group, label in [("files", "app.apk"), ("apps", "Example")]:
        args = ["workspace", group, "upload", str(path)]
        human = CliRunner().invoke(cli, args)
        assert human.exit_code == 0, human.output
        assert f"✓ Uploaded {label}" in human.stdout
        machine = CliRunner().invoke(cli, [*args, "--json"])
        data = json.loads(machine.stdout)
        assert (data[0] if isinstance(data, list) else data)["status"] == "uploaded"
