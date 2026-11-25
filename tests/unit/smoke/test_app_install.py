"""Opt-in Counter APK installation smoke tests; never operate on Portal."""

import asyncio

import pytest
from conftest import CLOUD_ID, SERIAL

from mobilerun.clients.local.apk import inspect_apk
from mobilerun.core.devices import AndroidDevice
from mobilerun.core.devices.cloud import CloudDevice


@pytest.fixture
def counter_apk(real_smoke_apk):
    path = real_smoke_apk
    if not path.is_file():
        pytest.skip(f"Counter smoke APK missing: {path}; set MOBILERUN_SMOKE_APK")
    package, _, _, _ = inspect_apk(path)
    if package != "ai.droidrun.counter":
        pytest.skip("Smoke APK is not the Counter app; refusing to modify another app")
    return path


def _has_counter(apps):
    return any(
        (app.get("package") or app.get("package_name") or app.get("packageName"))
        == "ai.droidrun.counter"
        for app in apps
    )


@pytest.mark.smoke_device
def test_local_counter_install(local_ready, counter_apk):
    async def run():
        device = AndroidDevice(serial=SERIAL)
        await device.connect()
        app = await device.install_apk(str(counter_apk), reinstall=True)
        assert app.package == "ai.droidrun.counter"
        try:
            await device.start_app(app.package)
            assert _has_counter(await device.list_apps())
            await device.uninstall_app(app.package)
        finally:
            await device.install_apk(str(counter_apk), reinstall=True)
        assert _has_counter(await device.list_apps())

    asyncio.run(run())


@pytest.mark.smoke_cloud
def test_cloud_counter_upload_install_and_uninstall(cloud_ready, counter_apk):
    async def run():
        device = CloudDevice(device_id=CLOUD_ID)
        await device.connect()
        app = await device.install_apk(counter_apk)
        assert app.package == "ai.droidrun.counter"
        try:
            await device.start_app(app.package)
            assert _has_counter(await device.list_apps())
            await device.uninstall_app(app.package)
        finally:
            await device.install_app(app)
        assert _has_counter(await device.list_apps())

    asyncio.run(run())
