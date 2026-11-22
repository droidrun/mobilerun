"""Built-in UI state providers are created lazily for device wrappers."""

import mobilerun.core.state.android as android_device
import mobilerun.core.state.ios as ios_device
import mobilerun.core.state.screenshot as visual_remote_device
from mobilerun.core.devices import (
    AndroidDevice,
    CloudDevice,
    IOSDevice,
    RecordingDevice,
    RemoteDevice,
    StealthDevice,
)
from mobilerun.core.state.android import AndroidStateProvider
from mobilerun.core.state.filters import DetailedFilter
from mobilerun.core.state.formatters import IndexedFormatter
from mobilerun.core.state.ios import IOSStateProvider
from mobilerun.core.state.options import StateProviderOptions
from mobilerun.core.state.screenshot import ScreenshotOnlyStateProvider


def test_platform_defaults_are_lazy_and_use_expected_provider(monkeypatch):
    made = []

    def android(driver, **kwargs):
        made.append(("android", kwargs))
        return AndroidStateProvider.__new__(AndroidStateProvider)

    def ios(driver, **kwargs):
        made.append(("ios", kwargs))
        return IOSStateProvider.__new__(IOSStateProvider)

    def screenshot(driver, **kwargs):
        made.append(("screenshot", kwargs))
        return ScreenshotOnlyStateProvider.__new__(ScreenshotOnlyStateProvider)

    monkeypatch.setattr(android_device, "AndroidStateProvider", android)
    monkeypatch.setattr(ios_device, "IOSStateProvider", ios)
    monkeypatch.setattr(visual_remote_device, "ScreenshotOnlyStateProvider", screenshot)
    devices = [
        AndroidDevice("a"),
        CloudDevice("c", api_key="fake"),
        IOSDevice("http://i"),
        RemoteDevice(url="http://v", device_id="v"),
    ]
    assert made == []
    from mobilerun_core_local.driver.ios import IOSPortalDriver

    devices[2]._driver = IOSPortalDriver("http://i")
    [device.state._resolved_provider for device in devices]
    assert [kind for kind, _ in made] == [
        "android",
        "android",
        "ios",
        "screenshot",
    ]
    assert made[0][1]["tree_filter"].__class__ is DetailedFilter
    assert made[0][1]["tree_formatter"].__class__ is IndexedFormatter


def test_provider_override_and_reset_to_default(monkeypatch):
    device = AndroidDevice("a")
    override = object()
    device._set_state_provider(override)
    assert device.state._resolved_provider is override
    device._set_state_provider(None)
    assert isinstance(device.state._resolved_provider, AndroidStateProvider)


def test_use_default_provider_passes_options(monkeypatch):
    args = []
    monkeypatch.setattr(
        android_device,
        "AndroidStateProvider",
        lambda d, **kw: args.append(kw) or object(),
    )
    device = AndroidDevice("a")
    device._reset_state_provider(StateProviderOptions(vision=False, normalized=True))
    _ = device.state._resolved_provider
    assert args[0]["vision_enabled"] is False
    assert args[0]["use_normalized"] is True


def test_wrappers_build_default_provider_against_outer_device():
    for wrapper_type in (RecordingDevice, StealthDevice):
        inner = AndroidDevice("a")
        provider = inner.state._resolved_provider
        wrapped = wrapper_type(inner)
        assert wrapped.state._resolved_provider is not provider
        assert wrapped.state._resolved_provider._device is wrapped
