"""Provider-backed UI state behavior without connecting to devices."""

import asyncio

import pytest

from mobilerun.config.settings import get_settings
from mobilerun.core.devices import (
    AndroidDevice,
    CloudDevice,
    RemoteDevice,
)
from mobilerun.errors import UnsupportedStateCapability


class Snapshot:
    elements = ["first", "second"]
    formatted_text = "indexed UI"

    def get_element_coords(self, index):
        return (index * 10, index * 20)

    def convert_point(self, x, y):
        return x + 1, y + 1


class FakeProvider:
    capabilities = {"element_index"}

    def __init__(self, device):
        self.device = device
        self.reads = 0

    async def get_state(self):
        self.reads += 1
        return Snapshot()

    async def get_final_state(self):
        return Snapshot()


class FullCapabilityProvider(FakeProvider):
    capabilities = {"element_index", "convert_point"}


DEVICE_CASES = [
    (AndroidDevice, ("serial-1",), {"use_tcp": True}),
    (
        CloudDevice,
        ("device-1",),
        {
            "display_id": 2,
            "api_key": "fake",
            "base_url": get_settings().MOBILERUN_CLOUD_BASE_URL,
        },
    ),
    (RemoteDevice, (), {"url": "http://visual-device", "device_id": "v1"}),
]


def patch_driver_init(monkeypatch, device_class):

    driver_class = device_class._backend
    calls = []

    def fake_init(self, *args, **kwargs):
        calls.append((args, kwargs.copy()))
        self._driver_args = args
        self._driver_kwargs = kwargs.copy()
        self.tapped = None
        self._client = None
        self._write_client = None
        self._serial = None

    async def fake_tap(self, x, y):
        self.tapped = (x, y)

    async def fake_ensure_connected(self):
        pass

    monkeypatch.setattr(driver_class, "__init__", fake_init)
    monkeypatch.setattr(driver_class, "tap", fake_tap, raising=False)
    monkeypatch.setattr(driver_class, "ensure_connected", fake_ensure_connected)
    return calls


@pytest.mark.parametrize("device_class,args,kwargs", DEVICE_CASES)
@pytest.mark.parametrize("provider_kind", ["class", "factory", "instance"])
def test_each_device_accepts_provider_without_changing_driver_args(
    monkeypatch, device_class, args, kwargs, provider_kind
):
    calls = patch_driver_init(monkeypatch, device_class)
    expected_args = (args, kwargs)
    if provider_kind == "class":
        provider = FakeProvider
    elif provider_kind == "factory":

        def provider(device):
            return FakeProvider(device)

    else:
        provider = FakeProvider(None)

    device = device_class(*args, **kwargs)
    if provider_kind == "instance":
        device._set_state_provider(provider)
    else:
        device._set_state_provider_factory(provider)

    assert calls == [expected_args]
    assert device._driver._driver_args == args
    assert device._driver._driver_kwargs == kwargs
    assert isinstance(device.state._resolved_provider, FakeProvider)
    if provider_kind == "instance":
        assert device.state._resolved_provider is provider
    else:
        assert device.state._resolved_provider.device is device


def test_refresh_caches_returned_snapshot_and_passthrough():
    async def check():
        device = AndroidDevice("serial")
        device._set_state_provider_factory(FakeProvider)
        snapshot = await device.state.refresh()
        assert device.state.current is snapshot
        assert device.state.current.formatted_text == "indexed UI"
        assert device.state.current.elements == ["first", "second"]
        assert device.state.current.get_element_coords(2) == (20, 40)
        assert device.state._resolved_provider.reads == 1

    asyncio.run(check())


def test_unsupported_capability_names_provider_and_capability():
    device = AndroidDevice("serial")
    device._set_state_provider_factory(FakeProvider)
    with pytest.raises(UnsupportedStateCapability, match="FakeProvider.*convert_point"):
        device.state._require("convert_point")


def test_device_without_default_has_clear_error():
    device = AndroidDevice("serial")

    def unavailable():
        from mobilerun.errors import DeviceNotReady

        raise DeviceNotReady("no state provider attached and no default")

    device._ui_state_view._default_provider = unavailable
    with pytest.raises(RuntimeError, match="no state provider attached and no default"):
        _ = device.state._resolved_provider


def test_attach_replaces_provider_and_clears_cached_state():
    async def check():
        device = AndroidDevice("serial")
        device._set_state_provider_factory(FakeProvider)
        old_state = await device.state.refresh()
        replacement = FullCapabilityProvider(device)
        device._set_state_provider(replacement)
        assert device.state._resolved_provider is replacement
        with pytest.raises(RuntimeError, match="not been refreshed"):
            _ = device.state.current
        assert await device.state.refresh() is not old_state

    asyncio.run(check())


def test_tap_element_uses_indexed_coordinates(monkeypatch):
    async def check():
        patch_driver_init(monkeypatch, AndroidDevice)
        device = AndroidDevice("serial")
        device._set_state_provider_factory(FakeProvider)
        await device.state.refresh()
        await device.tap_element(3)
        assert device._driver.tapped == (30, 60)

    asyncio.run(check())


def test_explicit_view_methods_reject_unsupported_capabilities():
    async def check():
        device = AndroidDevice("serial")
        device._set_state_provider_factory(FakeProvider)
        await device.state.refresh()
        names = dir(device.state)
        assert "formatted_text" not in names
        assert "elements" not in names
        assert "get_element_coords" not in names
        assert device.state.current.formatted_text == "indexed UI"
        with pytest.raises(UnsupportedStateCapability):
            device.state._require("convert_point")

    asyncio.run(check())
