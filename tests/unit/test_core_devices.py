"""Device facades retain the upstream driver construction contracts."""

import asyncio
import importlib
import subprocess
import sys

import pytest
from mobilerun_core_local.driver.android import AndroidDriver
from mobilerun_core_local.driver.base import DeviceDriver
from mobilerun_core_local.driver.ios import IOSDriver
from mobilerun_core_local.driver.visual_remote import VisualRemoteDriver


@pytest.mark.parametrize(
    ("name", "driver", "args", "kwargs", "platform"),
    [
        (
            "AndroidDevice",
            AndroidDriver,
            (),
            {"serial": "test", "portal_mode": "disabled"},
            "Android",
        ),
        ("IOSDevice", IOSDriver, ("http://localhost:8080",), {}, "iOS"),
        (
            "RemoteDevice",
            VisualRemoteDriver,
            (),
            {"url": "http://localhost:8090", "device_id": "test"},
            "VisualRemote",
        ),
    ],
)
def test_device_constructor_and_driver_compatibility(
    name, driver, args, kwargs, platform, monkeypatch
):
    devices = importlib.import_module("mobilerun.core.devices")
    device_class = getattr(devices, name)
    assert issubclass(device_class, devices.Device)
    device = device_class(*args, **kwargs)
    if name == "IOSDevice":
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "mobilerun.core.devices.ios.detect_ios_portal_kind",
            AsyncMock(return_value="legacy"),
        )
        monkeypatch.setattr(device, "_ensure_connected", AsyncMock())
        assert device._driver is None
        asyncio.run(device.connect())
        platform = "ios"
    assert isinstance(device._driver, driver)
    assert not isinstance(device, DeviceDriver)
    assert device.platform == platform
    for wrapper in (devices.RecordingDevice, devices.StealthDevice):
        wrapped = wrapper(device)
        assert wrapped._inner is device
        assert wrapped.platform == platform


def test_cloud_device_is_lazy_and_other_devices_do_not_require_cloud():
    script = """
import importlib.abc
import sys
class BlockCloud(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mobilerun_core_local.driver.cloud":
            raise ImportError("cloud deliberately unavailable")
sys.meta_path.insert(0, BlockCloud())
import mobilerun.core.devices as devices
assert "mobilerun_core_local.driver.cloud" not in sys.modules
assert devices.AndroidDevice
try:
    devices.CloudDevice
except ImportError as exc:
    assert "cloud deliberately unavailable" in str(exc)
else:
    raise AssertionError("cloud should be imported only on access")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_cloud_device_constructor_and_driver_compatibility():
    cloud = pytest.importorskip("mobilerun_core_local.driver.cloud")
    devices = importlib.import_module("mobilerun.core.devices")
    cls = devices.CloudDevice
    assert issubclass(cls, devices.Device)
    device = cls("test-device", api_key="test-key")
    assert isinstance(device._driver, cloud.CloudDriver)
    assert not isinstance(device, DeviceDriver)
    assert device.device_id == "test-device"
    assert device.platform == "Android"
    for wrapper in (devices.RecordingDevice, devices.StealthDevice):
        assert wrapper(device)._inner is device
