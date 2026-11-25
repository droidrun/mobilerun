"""Cloud status, action consent, resume billing, and discovery regressions."""

import asyncio
import importlib
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner
from mobilerun_sdk.types import DeviceRetrieveResponse

from mobilerun.cli.app import cli
from mobilerun.cli.devices.base import discovery, selection
from mobilerun.cli.devices.group import device
from mobilerun.clients.cloud import devices
from mobilerun.core.devices.base import Device
from mobilerun.core.extensions import Extension, action
from mobilerun.errors import BillingError
from mobilerun.models.billing import Billing

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
def test_bound_sms_send_requires_paid_billing_and_consent(
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
        with pytest.raises(BillingError, match="paid"):
            asyncio.run(devices.resume(CLOUD, billing=mode, client=client))
        client.devices.resume.assert_not_awaited()
    client.devices.retrieve.assert_awaited_once_with(CLOUD)


def test_resume_prompt_describes_existing_phone_billing(monkeypatch):
    from mobilerun.cli.base import menu, safety

    monkeypatch.setenv("MOBILERUN_BILLING", "included")
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(safety, "confirm_prompt_async", AsyncMock(return_value=False))
    result = CliRunner().invoke(device, ["cloud", "resume", CLOUD])
    assert "subscription" in result.output.lower()
    assert "quota" not in result.output


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
