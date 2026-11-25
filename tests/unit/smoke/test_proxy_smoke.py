"""Proxy smoke checks; the paid test owns and terminates its phone."""

import asyncio
from uuid import uuid4

import pytest
from conftest import CLOUD_ID, SERIAL

from mobilerun.clients.cloud.devices import list_devices
from mobilerun.clients.cloud.proxies import device_status, list_proxies
from mobilerun.core.devices import AndroidDevice
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.proxy import CloudRotatingProxy
from mobilerun.models import Billing, Plan
from tests.support.smoke_phone import close_smoke_phone

pytestmark = pytest.mark.smoke


def test_list_and_existing_status(real_env):
    async def run():
        assert (await list_proxies(billing=Billing.INCLUDED)).items is not None
        status = await device_status(CLOUD_ID, billing=Billing.INCLUDED)
        assert isinstance(status.connected, bool)

    asyncio.run(run())


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
