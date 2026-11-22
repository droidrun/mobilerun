import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from mobilerun.clients.cloud import devices
from mobilerun.config.settings import Settings
from mobilerun.errors import BillingError
from mobilerun.models import Billing, Plan


@pytest.mark.parametrize("plan", [Plan.MINUTE, "minute"])
@pytest.mark.parametrize("billing", [Billing.PAID, "paid"])
def test_public_enums_and_strings(billing, plan):
    create = AsyncMock(return_value="created")
    asyncio.run(
        devices.create(
            billing=billing,
            plan=plan,
            proxy={"connect": {"id": "existing"}},
            client=NS(devices=NS(create=create)),
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
