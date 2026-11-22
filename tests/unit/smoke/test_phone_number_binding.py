"""Paid number/phone binding with guaranteed cleanup and list verification."""

import asyncio

import pytest

from mobilerun.clients.cloud import devices, numbers
from mobilerun.core.devices import CloudDevice
from mobilerun.core.phone_number import CloudPhoneNumber
from mobilerun.core.proxy import CloudRotatingProxy
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.models import Billing, Plan

pytestmark = pytest.mark.smoke


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
            record_property("created_phone", phone._adb_device_id)
            await number.bind(phone)
            assert phone.number is number
            assert number.device_id == phone._adb_device_id
            await number.unbind()
            assert number.device_id is None
            with pytest.raises(DeviceOperationNotSupported):
                _ = phone.number
            proxy = CloudRotatingProxy(country="DE")
            status = await phone.set_proxy(proxy)
            assert status.connected and phone._proxy is proxy
            record_property("proxy_switch", "US->DE: active")
        finally:
            # Number cleanup is attempted even if phone cleanup subsequently fails.
            try:
                await number.release()
                record_property("released_number", number.id)
            finally:
                await phone.close()
                if phone._adb_device_id:
                    record_property("terminated_phone", phone._adb_device_id)
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
            if all(item.id != phone._adb_device_id for item in listed):
                break
            await asyncio.sleep(1)
        else:
            pytest.fail("Terminated phone is still listed")

    asyncio.run(exercise())
