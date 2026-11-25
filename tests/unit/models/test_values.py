import asyncio
import importlib
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from mobilerun.clients.cloud import devices
from mobilerun.config.settings import Settings
from mobilerun.errors import BillingError
from mobilerun.models import Billing, Plan
from mobilerun.utils.timestamps import to_api_timestamp


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(2025, 1, 2, tzinfo=timezone.utc), "2025-01-02T00:00:00Z"),
        (
            datetime(2025, 1, 2, 3, tzinfo=timezone(timedelta(hours=2))),
            "2025-01-02T01:00:00Z",
        ),
    ],
    ids=["utc", "offset"],
)
def test_api_timestamp_normalizes_to_utc(value, expected):
    assert to_api_timestamp(value) == expected


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        to_api_timestamp(datetime(2025, 1, 2))


def test_mailbox_wait_for_code_passes_api_timestamp_string():
    from mobilerun.core.mailbox.cloud import CloudMailbox

    async def poll(fetch, **kwargs):
        await fetch(None)
        return "1234"

    with (
        patch(
            "mobilerun.core.mailbox.cloud.mailboxes.otp", new_callable=AsyncMock
        ) as otp,
        patch("mobilerun.core.mailbox.cloud.wait_for_code", side_effect=poll),
    ):
        otp.return_value = None
        assert (
            asyncio.run(CloudMailbox("mailbox-id").wait_for_code(timeout=1)) == "1234"
        )
        otp.assert_awaited_once()
        after = otp.await_args.kwargs["after"]
        assert isinstance(after, str)
        assert after.endswith("Z")


def test_file_models_are_immutable_value_objects() -> None:
    from mobilerun.models.files import DeviceFile, WorkspaceFile

    now = datetime.now(timezone.utc)
    workspace = WorkspaceFile("id", "photo.jpg", "image/jpeg", 12, now)
    device = DeviceFile("/sdcard/photo.jpg", "photo.jpg", 12, now)
    assert workspace.name == device.name
    assert workspace.size_bytes == device.size_bytes
    with pytest.raises(FrozenInstanceError):
        workspace.name = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        device.path = "changed"  # type: ignore[misc]


def test_location_is_a_shared_value_model() -> None:
    from mobilerun.models import Location
    from mobilerun.models.location import Location as LocationModel

    assert Location is LocationModel
    assert Location(latitude=52.52, longitude=13.405).model_dump() == {
        "latitude": 52.52,
        "longitude": 13.405,
    }


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("ProxyStatus", "proxy"),
        ("RecordedAction", "recording"),
        ("CredentialMetadata", "credentials"),
        ("UIElement", "ui"),
        ("PaymentSnapshot", "checkout"),
    ],
)
def test_public_value_model_identity(name: str, module: str) -> None:
    models = importlib.import_module("mobilerun.models")
    home = importlib.import_module(f"mobilerun.models.{module}")
    assert getattr(models, name) is getattr(home, name)


def test_existing_core_exports_keep_model_identity() -> None:
    from mobilerun.core.proxy import ProxyStatus
    from mobilerun.core.state import UIElement
    from mobilerun.models import ProxyStatus as ModelProxyStatus
    from mobilerun.models import UIElement as ModelUIElement

    assert ProxyStatus is ModelProxyStatus
    assert UIElement is ModelUIElement


@pytest.mark.parametrize("plan", [Plan.MINUTE, "minute"])
@pytest.mark.parametrize("billing", [Billing.PAID, "paid"])
def test_public_enums_and_strings(billing, plan):
    create = AsyncMock(return_value="created")
    asyncio.run(
        devices.create(
            billing=billing,
            plan=plan,
            proxy={"connect": {"id": "existing"}},
            client=SimpleNamespace(devices=SimpleNamespace(create=create)),
        )
    )
    assert create.call_args.kwargs["billing"] == "minute"


@pytest.mark.parametrize("billing,plan", [("wrong", "auto"), ("paid", "wrong")])
def test_invalid_policy_values(billing, plan):
    with pytest.raises(BillingError, match="create.*given.*requires"):
        asyncio.run(
            devices.create(
                billing=billing, plan=plan, proxy={"connect": {"id": "existing"}}
            )
        )


def test_billing_setting_is_enum(monkeypatch):
    monkeypatch.setenv("MOBILERUN_BILLING", "paid")
    assert Settings().MOBILERUN_BILLING is Billing.PAID
