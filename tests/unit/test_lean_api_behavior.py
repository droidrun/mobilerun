"""Behavioral contracts for the lean constructor and observation API."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from mobilerun import AndroidDevice, Device, RecordingDevice, StealthDevice
from mobilerun.core.devices.components.wrapper import DriverDevice
from mobilerun.core.state import StateProvider, StateProviderOptions, UIState
from mobilerun.errors import DeviceNotReady, DeviceOperationNotSupported


@pytest.mark.parametrize("wrapper", [RecordingDevice, StealthDevice])
def test_explicit_wrapper_factory_replaces_inherited_provider(wrapper):
    inner = Device(state_provider=StateProvider(Device()))
    factory = Mock(side_effect=lambda device, options: StateProvider(device))
    options = StateProviderOptions(vision=True)
    outer = wrapper(inner, state_provider_factory=factory, state_options=options)
    assert outer.state._resolved_provider._device is outer
    factory.assert_called_once_with(outer, options)


@pytest.mark.parametrize("wrapper", [RecordingDevice, StealthDevice])
def test_wrapper_rejects_conflicting_explicit_provider_and_factory(wrapper):
    with pytest.raises(TypeError, match="state_provider or state_provider_factory"):
        wrapper(
            Device(),
            state_provider=StateProvider(Device()),
            state_provider_factory=lambda device, options: StateProvider(device),
        )


def test_state_controller_caches_returned_snapshot():
    snapshot = UIState(
        elements=[],
        formatted_text="ready",
        focused_text="",
        phone_state={},
        screen_width=100,
        screen_height=200,
    )
    provider = StateProvider(Device())
    provider.get_state = AsyncMock(return_value=snapshot)
    phone = Device(state_provider=provider)
    with pytest.raises(DeviceNotReady, match="refresh"):
        _ = phone.state.current
    assert asyncio.run(phone.state.refresh()) is snapshot
    assert phone.state.current is snapshot
    assert {name for name in dir(phone.state) if not name.startswith("_")} == {
        "refresh",
        "current",
        "capabilities",
    }


def test_driver_connect_is_idempotent_and_translates_unsupported():
    raw = Mock()
    raw.connect = AsyncMock(side_effect=NotImplementedError("no connect"))

    def factory(device, options):
        return StateProvider(device)

    phone = DriverDevice(raw, state_provider_factory=factory)
    with pytest.raises(DeviceOperationNotSupported, match="no connect"):
        asyncio.run(phone.connect())
    raw.connect.side_effect = None
    asyncio.run(phone.connect())
    asyncio.run(phone.connect())
    assert raw.connect.await_count == 2


def test_android_constructor_exposes_typed_state_configuration():
    import inspect

    names = inspect.signature(AndroidDevice).parameters
    assert {
        "state_provider",
        "state_provider_factory",
        "state_options",
        "operator",
    } <= names.keys()


def test_ios_cannot_cache_wrong_provider_before_detection(monkeypatch):
    from mobilerun import IOSDevice
    from mobilerun.core.state.android import AndroidStateProvider

    phone = IOSDevice("http://local")
    with pytest.raises(DeviceNotReady):
        _ = phone.state.capabilities
    monkeypatch.setattr(
        "mobilerun.core.devices.ios.detect_ios_portal_kind",
        AsyncMock(return_value="portal-http"),
    )
    monkeypatch.setattr(phone, "_ensure_connected", AsyncMock())
    asyncio.run(phone.connect())
    assert isinstance(phone.state._resolved_provider, AndroidStateProvider)


def test_android_text_boundary_matches_raw_driver_signature(monkeypatch):
    typed = []

    async def input_text(self, text, clear=False):
        typed.append((text, clear))
        return True

    monkeypatch.setattr(
        "mobilerun.core.devices.android.AndroidDriver.input_text", input_text
    )
    phone = AndroidDevice("fake")
    phone._driver.ensure_connected = AsyncMock()
    assert asyncio.run(phone.input_text("hello", clear=True))
    assert typed == [("hello", True)]
