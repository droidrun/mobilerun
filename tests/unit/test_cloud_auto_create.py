"""Auto-created cloud devices are owned and cleaned up."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.devices.base import Device
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.devices.pool import DevicePool


@pytest.fixture
def cloud(monkeypatch):
    monkeypatch.setenv("MOBILERUN_BILLING", "paid")
    from mobilerun_core_local.driver import cloud as driver

    devices = SimpleNamespace(
        retrieve_capabilities=AsyncMock(
            return_value=SimpleNamespace(capabilities=SimpleNamespace(telephony=False))
        ),
        create=AsyncMock(return_value=SimpleNamespace(id="created", name="created")),
        wait_ready=AsyncMock(),
        terminate=AsyncMock(),
        number=SimpleNamespace(
            retrieve=AsyncMock(return_value=SimpleNamespace(binding=None))
        ),
        actions=SimpleNamespace(tap=AsyncMock()),
    )
    client = SimpleNamespace(
        devices=devices, with_options=lambda **kw: client, close=AsyncMock()
    )
    monkeypatch.setattr(driver, "AsyncMobilerun", lambda **kw: client)
    return devices


def test_create_on_connect_and_cleanup(cloud, monkeypatch):
    monkeypatch.setenv("MOBILERUN_CLOUD_DEVICE_TYPE", "ios_phone")

    async def run():
        device = CloudDevice(api_key="test", name="example")
        assert device.device_id is None
        await asyncio.gather(device.connect(), device.connect())
        assert device.device_id == "created"
        cloud.create.assert_awaited_once_with(
            device_type="ios_phone",
            name="example",
            billing="auto",
            proxy={"connect": {"country": "us"}, "smart_ip": True},
        )
        await device.close()
        await device.close()
        cloud.terminate.assert_awaited_once_with("created")

    asyncio.run(run())


@pytest.mark.parametrize("keep", [False, True])
def test_context_and_keep(cloud, keep):
    async def run():
        async with CloudDevice(
            api_key="test", keep=keep, device_type="device_slot"
        ) as device:
            assert device.identifier == "created"
            assert device._keep is keep
        cloud.create.assert_awaited_once_with(
            device_type="device_slot",
            billing="auto",
            proxy={"connect": {"country": "us"}, "smart_ip": True},
        )
        assert cloud.terminate.await_count == (0 if keep else 1)

    asyncio.run(run())


def test_existing_id_is_never_owned(cloud):
    async def run():
        phone = CloudDevice("existing", api_key="test")
        from mobilerun.core.proxy import ProxyStatus

        phone._proxy_status = AsyncMock(return_value=ProxyStatus(False))
        async with phone:
            pass
        phone._proxy_status.assert_awaited_once()
        cloud.create.assert_not_awaited()
        cloud.terminate.assert_not_awaited()

    asyncio.run(run())


def test_pool_lifecycle(cloud):
    async def run():
        phone = CloudDevice(api_key="test")
        async with DevicePool([phone]):
            assert phone.device_id == "created"
        cloud.terminate.assert_awaited_once_with("created")

    asyncio.run(run())


def test_action_requires_awaited_construction(cloud):
    async def run():
        device = CloudDevice(api_key="test")
        with pytest.raises(RuntimeError, match="Await"):
            await device.tap(1, 2)
        cloud.create.assert_not_awaited()
        await device
        await device.tap(1, 2)
        assert cloud.actions.tap.await_args.args == ("created",)
        await device.close()

    asyncio.run(run())


def test_creation_failure_cleans_up(cloud):
    cloud.wait_ready.side_effect = RuntimeError("not ready")

    async def run():
        device = CloudDevice(api_key="test")
        with pytest.raises(RuntimeError, match="not ready"):
            await device
        cloud.terminate.assert_awaited_once_with("created")

    asyncio.run(run())


def test_auto_create_accepts_device_options(cloud):
    from mobilerun.core.operator import Operator

    operator = Operator()
    device = CloudDevice(api_key="test", operator=operator, name="test")
    assert device._operator is operator
    asyncio.run(device.connect())
    cloud.create.assert_awaited_once_with(
        device_type="android_cloud_phone",
        name="test",
        billing="auto",
        proxy={"connect": {"country": "us"}, "smart_ip": True},
    )


def test_documented_creation_options_reach_sdk_and_name_comes_from_server(cloud):
    cloud.create.return_value = SimpleNamespace(id="created", name="server-name")

    async def run():
        phone = await CloudDevice(
            api_key="test",
            name="requested",
            country="DE",
            locale="de-DE",
            timezone="Europe/Berlin",
            android_version=15,
            apps=["com.android.chrome"],
        )
        assert phone.name == "server-name"
        cloud.create.assert_awaited_once_with(
            device_type="android_cloud_phone",
            billing="auto",
            name="requested",
            body_country="DE",
            locale="de-DE",
            timezone="Europe/Berlin",
            android_version=15,
            apps=["com.android.chrome"],
            proxy={"connect": {"country": "us"}, "smart_ip": True},
        )
        await phone.close()

    asyncio.run(run())


def test_template_is_forwarded_without_request_apps_or_version(cloud):
    phone = CloudDevice(api_key="test", name="prepared", template="onboarding")
    asyncio.run(phone.connect())
    assert cloud.create.await_args.kwargs["template"] == "onboarding"
    assert "apps" not in cloud.create.await_args.kwargs
    assert "android_version" not in cloud.create.await_args.kwargs


def test_pool_explicit_close(cloud):
    async def run():
        pool = DevicePool([CloudDevice(api_key="test")])
        await pool.__aenter__()
        await pool.close()
        cloud.terminate.assert_awaited_once_with("created")

    asyncio.run(run())


def test_pool_context_cleans_up_if_another_device_cannot_connect(cloud):
    class Broken(Device):
        @property
        def needs_initialization(self):
            return False

        async def _ensure_connected(self):
            raise RuntimeError("connect failed")

        async def close(self):
            pass

    async def run():
        with pytest.raises(ExceptionGroup):
            async with DevicePool([CloudDevice(api_key="test"), Broken()]):
                pass
        cloud.terminate.assert_awaited_once_with("created")

    asyncio.run(run())


def test_owned_cloud_phone_is_terminated_through_wrapper(cloud):
    from mobilerun.core.devices.recording import RecordingDevice

    async def run():
        async with RecordingDevice(
            CloudDevice(api_key="test", billing="paid", plan="minute")
        ) as device:
            assert device.identifier == "created"
        cloud.create.assert_awaited_once_with(
            device_type="android_cloud_phone",
            billing="minute",
            proxy={"connect": {"country": "us"}, "smart_ip": True},
        )
        cloud.terminate.assert_awaited_once_with("created")

    asyncio.run(run())
