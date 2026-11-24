"""Cloud phone rename command contracts without network access."""

import importlib
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from mobilerun.cli.devices.group import device
from mobilerun.config.settings import get_settings
from mobilerun.errors import CloudPermissionDenied

DEVICE = "11111111-1111-1111-1111-111111111111"


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
            DEVICE,
            "  New name  ",
            "--json",
            "--base-url",
            "https://example.test",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(result.stdout) == {"id": DEVICE, "name": "New name"}
    billing = get_settings().MOBILERUN_BILLING
    assert calls == [{"base_url": "https://example.test", "billing": billing}]
    endpoint.assert_awaited_once_with(
        DEVICE, "New name", billing=billing, client=opened
    )


def test_rename_interactive_prompt(rename_setup, monkeypatch):
    module, endpoint, _, _ = rename_setup
    monkeypatch.setattr(module, "is_interactive", lambda: True)
    monkeypatch.setattr(module, "text_prompt_async", AsyncMock(return_value=" Phone "))
    result = CliRunner().invoke(device, ["cloud", "rename", DEVICE])
    assert result.exit_code == 0, (result.output, result.exception)
    assert f"✓ Renamed {DEVICE} to Phone" in result.output
    assert endpoint.await_args.args == (DEVICE, "Phone")


def test_rename_empty_prompt_cancels(rename_setup, monkeypatch):
    module, endpoint, _, _ = rename_setup
    monkeypatch.setattr(module, "is_interactive", lambda: True)
    monkeypatch.setattr(module, "text_prompt_async", AsyncMock(return_value="  "))
    result = CliRunner().invoke(device, ["cloud", "rename", DEVICE])
    assert result.exit_code == 0
    assert result.stderr == "Cancelled.\n"
    endpoint.assert_not_awaited()


@pytest.mark.parametrize("args", [[DEVICE, "  "], [DEVICE]])
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
        choices, "cloud_devices", lambda ctx: [SelectChoice(DEVICE, "Phone")]
    )
    monkeypatch.setattr(
        pickers, "select_prompt", lambda label, entries: entries[0].value
    )
    result = CliRunner().invoke(device, ["cloud", "rename"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert endpoint.await_args.args == (DEVICE, "New")


@pytest.mark.parametrize("json_option", [[], ["--json"]])
def test_rename_cloud_error_is_reported_on_stderr(rename_setup, json_option):
    _, endpoint, _, _ = rename_setup
    error = CloudPermissionDenied(operation="Rename cloud phone", status=403)
    endpoint.side_effect = error
    result = CliRunner().invoke(
        device, ["cloud", "rename", DEVICE, "New", *json_option]
    )
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
            devices.set_name(DEVICE, "name", billing="no_commerce", client=client)
        )
    assert "sensitive-value" not in str(caught.value)
    client.close.assert_not_awaited()
