"""Existing cloud phone checks and explicitly opted-in billable lifecycle tests."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from mobilerun.clients.cloud.api_keys import validate_api_key
from mobilerun.core.credentials import DictCredentials
from mobilerun.core.devices import CloudDevice
from mobilerun.errors import (
    CloudAuthRequired,
    CloudDeviceNotFound,
    LocalCredentialsNotAllowed,
)
from mobilerun.errors.cloud import CloudAuthenticationError
from mobilerun.models import Billing, Plan
from tests.support.smoke_phone import close_smoke_phone
from tests.support.smoke_settings import CLOUD_ID


@pytest.mark.smoke_cloud
def test_cloud_info_and_wait_ready(cloud_ready):
    async def run():
        device = CloudDevice(CLOUD_ID)
        assert (await device.info()).id == CLOUD_ID
        await device.wait_ready(timeout=30)

    asyncio.run(run())


@pytest.mark.smoke_cloud
def test_cloud_rejects_local_secrets(cloud_ready):
    async def run():
        device = CloudDevice(CLOUD_ID)
        await DictCredentials({"test": "not-a-real-secret"}).bind(device)
        with pytest.raises(LocalCredentialsNotAllowed):
            await device.do("This must never be submitted")

    asyncio.run(run())


@pytest.mark.smoke_cloud
def test_paid_cloud_create_lifecycle(cloud_ready, record_property):
    async def run():
        name = f"SDK lifecycle smoke {uuid4().hex}"
        record_property("device_name", name)
        device = CloudDevice(name=name, billing=Billing.PAID, plan=Plan.MINUTE)
        try:
            await device
            info = await device.info()
            assert info.id == device.cloud_id
            assert info.type == "android_cloud_phone"
            await device.stop()
            await device.resume()
            await device.wait_ready()
            await device.reboot()
            await device.wait_ready()
            await device.reset(mode="identity")
            await device.wait_ready()
        finally:
            await close_smoke_phone(device, name)

    asyncio.run(run())


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


def test_unknown_cloud_device_has_clear_error(real_env):
    device = CloudDevice("00000000-0000-0000-0000-000000000000")
    with pytest.raises(CloudDeviceNotFound):
        asyncio.run(device.do("noop", max_steps=1))


@pytest.mark.smoke_cloud
def test_status_json_excludes_connection_credentials(cloud_ready):
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("mobilerun")),
            "device",
            "cloud",
            "status",
            CLOUD_ID,
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        pytest.fail("Cloud status command failed; output withheld")
    fields = json.loads(result.stdout).keys()
    assert {"id", "state"} <= fields
    assert not {"streamToken", "stream_token", "streamUrl", "stream_url"}.intersection(
        fields
    )


@pytest.mark.smoke_cloud
@pytest.mark.parametrize(
    "args",
    [["device", "list", "--cloud"], ["account", "status"]],
)
def test_cloud_cli(cloud_ready, args):
    result = subprocess.run(
        [str(Path(sys.executable).with_name("mobilerun")), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (
        result.returncode == 0
    ), "Cloud CLI failed (output withheld to protect credentials)"
    if args[:2] == ["device", "list"]:
        assert CLOUD_ID in result.stdout


@pytest.mark.smoke_cloud
def test_missing_cloud_key(real_env, monkeypatch):
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    with pytest.raises(CloudAuthRequired):
        CloudDevice(device_id=CLOUD_ID)


@pytest.mark.smoke_cloud
def test_cloud_ping_reports_unsupported(cloud_ready):
    result = subprocess.run(
        [str(Path(sys.executable).with_name("mobilerun")), "device", "ping", CLOUD_ID],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "supports Android targets only" in result.stderr


def test_real_api_key_is_valid(real_env):
    if not os.environ.get("MOBILERUN_API_KEY"):
        pytest.skip("MOBILERUN_API_KEY is not set")
    asyncio.run(validate_api_key(os.environ["MOBILERUN_API_KEY"], billing="included"))


def test_bad_api_key_is_invalid(real_env):
    with pytest.raises(CloudAuthenticationError):
        asyncio.run(
            validate_api_key("obviously-invalid-mobilerun-api-key", billing="included")
        )
