"""Framework roots dispatch to raw backends without sharing incompatible APIs."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.devices import AndroidDevice, IOSDevice, RemoteDevice
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.state.elements import _parse_bounds


@pytest.mark.parametrize(
    "device",
    [
        AndroidDevice("fake"),
        IOSDevice("http://localhost"),
        RemoteDevice("http://localhost"),
    ],
)
def test_root_io_uses_its_backend_and_live_capabilities(device):
    if device._driver is None:
        from mobilerun_core_local.driver.ios import IOSPortalDriver

        device._driver = IOSPortalDriver("http://localhost")
    device._driver.tap = AsyncMock()
    device._driver.ensure_connected = AsyncMock()
    device._driver.supported = {"tap"}
    asyncio.run(device.tap(4, 8))
    device._driver.tap.assert_awaited_once_with(4, 8)
    assert "tap" in device.supported_operations


def test_cloud_io_syncs_created_id_and_current_clients():
    device = CloudDevice(api_key="test")
    device.device_id = "created"
    device._ready = True
    tap = AsyncMock()
    device._write_client = SimpleNamespace(
        devices=SimpleNamespace(actions=SimpleNamespace(tap=tap))
    )
    asyncio.run(device.tap(4, 8))
    assert tap.await_args.args == ("created",)
    assert device._driver._client is device._client
    assert device._driver._write_client is device._write_client


def test_pending_android_action_never_reaches_raw_backend():
    device = AndroidDevice(avd="fake")
    device._driver.tap = AsyncMock()
    with pytest.raises(RuntimeError, match="Await AndroidDevice"):
        asyncio.run(device.tap(4, 8))
    device._driver.tap.assert_not_awaited()


@pytest.mark.parametrize("value", ["1,2,3", "1,2,3,4,5", "a,2,3,4"])
def test_bounds_parser_rejects_malformed_geometry(value):
    with pytest.raises(ValueError):
        _parse_bounds(value)


def test_bounds_parser_has_exact_four_coordinate_shape():
    assert _parse_bounds("1,2,3,4") == (1, 2, 3, 4)


def test_cloud_rejects_unknown_device_type_before_create():
    with pytest.raises(ValueError, match="Invalid cloud device type"):
        CloudDevice(api_key="test", device_type="not-a-device-type")
