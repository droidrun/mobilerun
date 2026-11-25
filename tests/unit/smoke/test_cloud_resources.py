"""Opt-in resource smoke tests; each case retains its own billing and cleanup rules."""

import asyncio
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from mobilerun_sdk import APIStatusError

from mobilerun.clients.cloud import devices, mailboxes, numbers, proxies
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.clients.cloud.devices import list_devices
from mobilerun.clients.cloud.proxies import device_status, list_proxies
from mobilerun.config.settings import get_settings
from mobilerun.core.devices import AndroidDevice, CloudDevice
from mobilerun.core.mailbox import CloudMailbox
from mobilerun.core.phone_number import CloudPhoneNumber
from mobilerun.core.proxy import CloudRotatingProxy
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.models.billing import Billing, Plan
from tests.support.smoke_phone import close_smoke_phone
from tests.support.smoke_settings import CLOUD_ID, SERIAL


@pytest.mark.smoke_cloud
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


@pytest.mark.smoke_cloud
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


def run_check(coro):
    try:
        return asyncio.run(coro)
    except APIStatusError as exc:
        pytest.fail(
            f"Communications API returned HTTP {exc.status_code}; response omitted"
        )


@pytest.mark.smoke
def test_free_lists_countries_and_capacity(real_env):
    async def exercise():
        await numbers.list_numbers(billing="no_commerce")
        countries = await numbers.countries(billing="no_commerce")
        assert countries.data.items
        await numbers.capacity(countries.data.items[0].country, billing="no_commerce")
        await mailboxes.list_mailboxes(billing="no_commerce")
        await mailboxes.capacity(billing="no_commerce")

    run_check(exercise())


@pytest.mark.smoke
def test_paid_number_create_release_and_sms(real_env, record_property):
    async def exercise():
        countries = await numbers.countries(billing="included")
        country = next(
            (item.country for item in countries.data.items if item.in_stock), None
        )
        if not country:
            pytest.skip("No phone numbers in stock")
        number = await CloudPhoneNumber(
            country=country, billing="paid", label="mobilerun-comms-smoke"
        )
        record_property("created_number", number.id)
        try:
            # Poll for provisioning without creating any further resources.
            info = None
            for _ in range(30):
                info = (await numbers.retrieve(number.id, billing="included")).data
                if info.state != "provisioning":
                    break
                await asyncio.sleep(2)
            if info.state != "active" or not info.actions.send or not info.phone_number:
                record_property(
                    "sms_flow", "unavailable: number not active/send-enabled"
                )
                return
            after = datetime.now(timezone.utc)
            code = str(secrets.randbelow(900000) + 100000)
            await number.send(info.phone_number, "Mobilerun test verification: " + code)
            received = await number.wait_for_code(timeout=60, after=after)
            if received != code:
                pytest.fail("SMS verification did not match; values omitted")
            record_property("sms_flow", "verified")
        finally:
            await number.release()
            remaining = next(
                (
                    item
                    for item in await numbers.list_numbers(billing="included")
                    if item.id == number.id
                ),
                None,
            )
            assert (
                remaining is None
                or remaining.cancel_at_period_end
                or remaining.state in ("expired", "failed")
            )
            record_property("released_number", number.id)

    run_check(exercise())


@pytest.mark.smoke
def test_paid_mailbox_create_release(real_env, record_property):
    async def exercise():
        mailbox = await CloudMailbox(billing="paid", label="mobilerun-comms-smoke")
        record_property("created_mailbox", mailbox.id)
        try:
            await mailboxes.retrieve(mailbox.id, billing="included")
            # Receiving real email requires an external sender, which this SDK lacks.
            record_property(
                "email_flow", "not run: no authorized outbound email transport"
            )
        finally:
            await mailbox.release()
            remaining = next(
                (
                    item
                    for item in await mailboxes.list_mailboxes(billing="included")
                    if item.id == mailbox.id
                ),
                None,
            )
            assert (
                remaining is None
                or remaining.cancel_at_period_end
                or remaining.status in ("archived", "billing_error")
            )
            record_property("released_mailbox", mailbox.id)

    run_check(exercise())


@pytest.mark.smoke
def test_existing_mailbox_messages(real_env):
    async def exercise():
        existing = await mailboxes.list_mailboxes(billing="no_commerce")
        if not existing:
            pytest.skip("No existing mailbox available")
        result = await mailboxes.messages(existing[0].id, billing="no_commerce")
        assert isinstance(result, list)

    asyncio.run(exercise())


@pytest.mark.smoke_cloud
def test_number_included_all_countries(real_env):
    if not os.getenv("MOBILERUN_API_KEY"):
        pytest.skip("Cloud API key is not set")
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("mobilerun")),
            "number",
            "included",
            "--billing",
            "no_commerce",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, "Cloud CLI failed (output withheld)"
    entries = json.loads(result.stdout)
    assert isinstance(entries, list)
    assert all(set(entry) == {"country", "remaining"} for entry in entries)
    assert all(
        isinstance(entry["country"], str)
        and (entry["remaining"] is None or isinstance(entry["remaining"], int))
        for entry in entries
    )


@pytest.mark.smoke
def test_paid_number_bind_unbind_and_release(real_env, record_property):
    async def exercise():
        countries = await numbers.countries(billing=Billing.NO_COMMERCE)
        country = next(
            (item.country for item in countries.data.items if item.in_stock), None
        )
        if country is None:
            pytest.skip("No phone numbers in stock")
        number = await CloudPhoneNumber(
            country=country, billing=Billing.PAID, label="mobilerun-binding-smoke"
        )
        record_property("created_number", number.id)
        started = asyncio.get_running_loop().time()
        phone = CloudDevice(
            billing=Billing.PAID, plan=Plan.MINUTE, name="mobilerun-binding-smoke"
        )
        try:
            await number.wait_ready()
            record_property(
                "number_ready_seconds",
                f"{asyncio.get_running_loop().time() - started:.1f}",
            )
            await phone
            phone_id = phone.cloud_id
            assert phone_id is not None
            record_property("created_phone", phone_id)
            await number.bind(phone)
            assert phone.number is number
            assert number.device_id == phone_id
            async with CloudDevice(phone_id, billing=Billing.NO_COMMERCE) as observer:
                assert observer.number.id == number.id
                assert observer.actions.get("device.open_url")
            async with cloud_client(billing=Billing.NO_COMMERCE) as client:
                remote = await devices.number(
                    phone_id, billing=Billing.NO_COMMERCE, client=client
                )
            assert remote.binding and remote.binding.number_id == number.id
            await number.unbind()
            assert number.device_id is None
            with pytest.raises(DeviceOperationNotSupported):
                _ = phone.number
            proxy = CloudRotatingProxy(country="DE")
            await proxy.bind(phone, force=True)
            assert (await proxy.status()).connected and phone.extensions[
                "proxy"
            ] is proxy
            record_property("proxy_switch", "US->DE: active")
        finally:
            # Number cleanup is attempted even if phone cleanup subsequently fails.
            try:
                await number.release()
                record_property("released_number", number.id)
            finally:
                terminated_id = phone.cloud_id
                await phone.close()
                if terminated_id is not None:
                    record_property("terminated_phone", terminated_id)
        remaining = next(
            (
                item
                for item in await numbers.list_numbers(billing=Billing.NO_COMMERCE)
                if item.id == number.id
            ),
            None,
        )
        assert (
            remaining is None
            or remaining.cancel_at_period_end
            or remaining.state in ("expired", "failed")
        )
        for _ in range(30):
            listed = await devices.list_devices(billing=Billing.NO_COMMERCE)
            if all(item.id != phone_id for item in listed):
                break
            await asyncio.sleep(1)
        else:
            pytest.fail("Terminated phone is still listed")

    asyncio.run(exercise())


@pytest.mark.smoke_cloud
def test_connected_phone_advertises_server_proxy_capability(cloud_ready):
    async def run():
        info = await devices.capabilities(CLOUD_ID, billing=Billing.NO_COMMERCE)
        async with CloudDevice(CLOUD_ID, billing=Billing.NO_COMMERCE) as phone:
            assert (
                "proxy_switch" in phone.supported_operations
            ) == info.capabilities.proxy
            assert await phone.screenshot()

    asyncio.run(run())


@pytest.mark.smoke
def test_list_and_existing_status(real_env):
    async def run():
        assert (await list_proxies(billing=Billing.INCLUDED)).items is not None
        status = await device_status(CLOUD_ID, billing=Billing.INCLUDED)
        assert isinstance(status.connected, bool)

    asyncio.run(run())


@pytest.mark.smoke
@pytest.mark.smoke_device
def test_local_proxy_cleared(real_env):
    async def run():
        device = AndroidDevice(serial=SERIAL, portal_mode="disabled")
        from mobilerun.core.proxy import LocalProxy

        proxy = LocalProxy("127.0.0.1", 8888)
        try:
            if (await device._proxy_target.status()).connected:
                pytest.skip("Preserving the emulator's existing proxy")
            await proxy.bind(device)
            assert device.proxy is proxy
            assert proxy.device is device
            assert (await proxy.status()).connected
            await proxy.unbind()
            assert not device.extensions
            assert not (await device._proxy_target.status()).connected
        finally:
            await device.close()

    asyncio.run(run())


@pytest.mark.smoke
def test_paid_create_with_proxy(real_env, record_property):
    async def run():
        before = await list_proxies(
            billing=Billing.INCLUDED, country="de", page_size=100
        )
        if not any(
            item.status == "active" and item.type == "residential"
            for item in before.items
        ):
            pytest.skip("No active German residential proxy available for safe reuse")
        name = f"proxy smoke {uuid4().hex}"
        record_property("device_name", name)
        device = CloudDevice(
            proxy=CloudRotatingProxy(country="DE"),
            billing=Billing.PAID,
            plan=Plan.MINUTE,
            name=name,
        )
        try:
            await device
            record_property("device_id", device.cloud_id)
            assert (await device.proxy.status()).connected
        finally:
            await close_smoke_phone(device, name)
            device_id = device.cloud_id
            if device_id:
                for _ in range(30):
                    listed = await list_devices(billing=Billing.INCLUDED)
                    if all(item.id != device_id for item in listed):
                        break
                    await asyncio.sleep(1)
                else:
                    pytest.fail(f"Terminated phone {device_id} is still listed")
                record_property("released_device", device_id)
        before_ids = {item.id for item in before.items}
        new = []
        for _ in range(30):
            after = await list_proxies(
                billing=Billing.INCLUDED, country="de", page_size=100
            )
            new = [item for item in after.items if item.id not in before_ids]
            if all(item.status == "ended" for item in new):
                break
            await asyncio.sleep(1)
        for item in new:
            record_property("new_proxy", f"{item.id}:{item.status}")
        assert all(item.status == "ended" for item in new), "New proxy is not released"

    asyncio.run(run())
