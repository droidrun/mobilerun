from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from mobilerun.cli.devices.group import device
from mobilerun.errors import DeviceOperationNotSupported


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


@pytest.mark.parametrize(
    ("args", "feature"),
    [
        (["input", "clipboard", "get"], "get_clipboard"),
        (["input", "clipboard", "set", "text"], "set_clipboard"),
        (["settings", "language", "set", "de-DE"], "set_language"),
    ],
)
def test_cloud_only_environment_rejects_local_before_connect(
    monkeypatch, args, feature
):
    from mobilerun.cli.devices.base import selection as environment
    from mobilerun.core.devices.base import Device

    connect = AsyncMock(return_value=Device())
    monkeypatch.setattr(environment, "connect_target", connect)
    result = CliRunner().invoke(device, [*args, "--device", "emulator-5554"])
    assert result.exit_code == 1
    assert result.output.strip() == f"Error: Device doesn't support {feature}"
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
