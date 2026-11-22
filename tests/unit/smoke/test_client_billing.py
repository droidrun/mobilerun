"""Real client billing checks; the paid phone is always terminated."""

import asyncio
import time

import pytest

from mobilerun.clients.cloud import devices, mailboxes, numbers, proxies
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.config.settings import get_settings
from mobilerun.models.billing import Billing, Plan

pytestmark = pytest.mark.smoke_cloud


def test_free_clients_no_commerce(real_env):
    async def run():
        async with cloud_client(billing=Billing.NO_COMMERCE) as client:
            assert isinstance(
                await devices.list_devices(client=client, billing=Billing.NO_COMMERCE),
                list,
            )
            assert isinstance(
                await numbers.list_numbers(client=client, billing=Billing.NO_COMMERCE),
                list,
            )
            assert isinstance(
                await mailboxes.list_mailboxes(
                    client=client, billing=Billing.NO_COMMERCE
                ),
                list,
            )
        assert (
            await proxies.list_proxies(billing=Billing.NO_COMMERCE)
        ).items is not None

    asyncio.run(run())


def test_paid_minute_phone_released(real_env, record_property):
    async def run():
        listed = await proxies.list_proxies(billing=Billing.NO_COMMERCE, page_size=100)
        proxy = next((item for item in listed.items if item.status == "active"), None)
        if proxy is None:
            pytest.skip(
                "No existing active Connect proxy; do not purchase a proxy for this smoke"
            )
        started = time.monotonic()
        async with cloud_client(billing=Billing.PAID) as client:
            phone = await devices.create(
                client=client,
                billing=Billing.PAID,
                plan=Plan.MINUTE,
                proxy={"connect": {"id": proxy.id}},
                device_type=get_settings().MOBILERUN_CLOUD_DEVICE_TYPE,
                name="client-billing-minute-smoke",
            )
            record_property("created_device", phone.id)
            try:
                await devices.wait_ready(
                    phone.id, client=client, billing=Billing.PAID, timeout=180
                )
                assert (
                    await devices.retrieve(
                        phone.id, client=client, billing=Billing.NO_COMMERCE
                    )
                ).id == phone.id
            finally:
                await devices.terminate(
                    phone.id, client=client, billing=Billing.NO_COMMERCE
                )
                record_property("terminated_device", phone.id)
                record_property("elapsed_seconds", round(time.monotonic() - started, 2))
                record_property(
                    "billing_plan",
                    "minute; actual monetary debit not exposed by this API",
                )

    asyncio.run(run())
