"""Paid smoke: creates and terminates a billable cloud phone; opt in explicitly."""

import asyncio

import pytest

from mobilerun.core.devices.cloud import CloudDevice


@pytest.mark.smoke
def test_auto_create_and_terminate(real_env, record_property):
    async def run():
        device = CloudDevice(billing="paid", plan="minute")
        async with device:
            assert device._adb_device_id
            record_property("created_device", device._adb_device_id)
        assert not device._owned
        record_property("terminated_device", device._adb_device_id)

    asyncio.run(run())
