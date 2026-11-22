"""Portable operations and backend-specific handles stay separate."""

import asyncio
from typing import assert_type

import pytest
from mobilerun_core_local.driver.base import DeviceDriver, PlatformUnsupportedError

from mobilerun.core.devices import RecordingDevice, StealthDevice
from mobilerun.core.devices.base import Device
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.devices.components.wrapper import DeviceWrapper, DriverDevice
from mobilerun.core.devices.pool import DevicePool
from mobilerun.core.state.options import screenshot_state_provider
from mobilerun.errors import DeviceOperationNotSupported


@pytest.mark.parametrize(
    "name",
    ["set_clipboard", "get_clipboard", "set_language"],
)
def test_environment_surface_is_portable(name: str) -> None:
    for cls in (Device, DeviceWrapper, RecordingDevice, StealthDevice, DevicePool):
        assert getattr(cls, name) is not None
    assert getattr(CloudDevice, name) is not None


def test_wrappers_preserve_typed_backend_handle() -> None:
    cloud = CloudDevice("phone", api_key="test")
    wrapped = RecordingDevice(StealthDevice(cloud))
    assert_type(wrapped._inner._inner, CloudDevice)
    assert wrapped._inner._inner is cloud
    assert wrapped._unwrap() is cloud


def test_local_number_and_unbound_mailbox_explain_unavailability() -> None:
    with pytest.raises(DeviceOperationNotSupported, match="No sms extension"):
        _ = Device().number
    with pytest.raises(DeviceOperationNotSupported, match="No email extension"):
        _ = Device().mailbox


def test_cloud_resources_work_through_wrappers_and_pool(monkeypatch) -> None:
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mobilerun.clients.cloud import devices, mailboxes, numbers
    from mobilerun.core.mailbox import CloudMailbox
    from mobilerun.core.phone_number import CloudPhoneNumber

    async def run() -> None:
        phone = CloudDevice("phone", api_key="test")
        phone._cloud_telephony = True
        number = CloudPhoneNumber("number", api_key="test")
        mailbox = CloudMailbox("mailbox", api_key="test")
        monkeypatch.setattr(devices, "bind_number", AsyncMock())
        await number.bind(phone)
        await mailbox.bind(phone)
        now = datetime.now(timezone.utc)
        sms = {
            "id": "message",
            "createdAt": now,
            "occurredAt": now,
            "direction": "inbound",
            "status": "received",
            "detectedSender": "sender",
            "body": "Code 123456",
        }
        monkeypatch.setattr(
            numbers,
            "messages",
            AsyncMock(return_value=[numbers.MessageItem.model_validate(sms)]),
        )
        result = numbers.MessageSendResponse.model_validate({"data": sms})
        monkeypatch.setattr(numbers, "send", AsyncMock(return_value=result))
        monkeypatch.setattr(
            mailboxes,
            "otp",
            AsyncMock(
                return_value=SimpleNamespace(data=SimpleNamespace(code="654321"))
            ),
        )
        wrapper = DeviceWrapper(phone)
        assert await wrapper.number.wait_for_code() == "123456"
        assert len(await wrapper.number.messages()) == 1
        assert await wrapper.number.send("recipient", "message") == result
        assert await wrapper.mailbox.wait_for_code() == "654321"
        pool = DevicePool([wrapper])
        assert pool.number == [number]
        assert pool.mailbox == [mailbox]

    asyncio.run(run())


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("tap", (1, 2)),
        ("swipe", (1, 2, 3, 4)),
        ("drag", (1, 2, 3, 4)),
        ("input_text", ("hello",)),
        ("screenshot", ()),
        ("start_app", ("app",)),
        ("stop_app", ("app",)),
        ("uninstall_app", ("app",)),
        ("list_apps", ()),
        ("press_button", ("home",)),
        ("press_key_code", (3,)),
        ("_get_ui_tree", ()),
        ("_get_date", ()),
        ("_ensure_connected", ()),
    ],
)
def test_base_unsupported_operations_have_framework_error(method, args) -> None:
    with pytest.raises(DeviceOperationNotSupported):
        asyncio.run(getattr(Device(), method)(*args))


@pytest.mark.parametrize("error", [NotImplementedError, PlatformUnsupportedError])
@pytest.mark.parametrize("operation", ["tap", "stop_app", "_get_ui_tree"])
def test_raw_adapter_translates_unsupported(error, operation) -> None:
    class UnsupportedDriver(DeviceDriver):
        async def tap(self, x: int, y: int) -> None:
            raise error("backend unsupported")

        async def stop_app(self, package: str) -> str:
            raise error("backend unsupported")

        async def get_ui_tree(self) -> dict[str, object]:
            raise error("backend unsupported")

    device = DriverDevice(
        UnsupportedDriver(), state_provider_factory=screenshot_state_provider
    )
    args = {"tap": (1, 2), "stop_app": ("app",), "_get_ui_tree": ()}[operation]
    with pytest.raises(DeviceOperationNotSupported, match="backend unsupported"):
        asyncio.run(getattr(device, operation)(*args))


@pytest.mark.parametrize(
    "method,args",
    [("list_apps", ()), ("_get_date", ()), ("_get_ui_tree", ())],
)
def test_visual_remote_does_not_report_empty_unsupported_results(method, args) -> None:
    from mobilerun.core.devices.remote import RemoteDevice

    with pytest.raises(DeviceOperationNotSupported):
        asyncio.run(getattr(RemoteDevice("http://localhost:1"), method)(*args))
