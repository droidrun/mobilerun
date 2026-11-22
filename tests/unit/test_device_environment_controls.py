import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.devices.base import Device
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.devices.pool import DevicePool
from mobilerun.errors import DeviceOperationNotSupported


def test_default_environment_is_unsupported():
    with pytest.raises(DeviceOperationNotSupported):
        asyncio.run(Device().open_url("https://example.com"))


def test_cloud_environment_calls_sdk():
    async def run():
        device = CloudDevice("abc", api_key="test")
        resources = SimpleNamespace(
            location=SimpleNamespace(set=AsyncMock(), reset=AsyncMock()),
            language=SimpleNamespace(set=AsyncMock()),
            timezone=SimpleNamespace(set=AsyncMock()),
            clipboard=SimpleNamespace(
                set=AsyncMock(),
                get=AsyncMock(return_value=SimpleNamespace(text="hello")),
            ),
            deep_link=SimpleNamespace(execute_deep_link=AsyncMock()),
        )
        device._write_client = cast(Any, SimpleNamespace(devices=resources))
        device._client = device._write_client
        await device.set_location(1.0, 2.0)
        await device.reset_location()
        await device.set_language("de-DE")
        await device.set_timezone("Europe/Berlin")
        await device.set_clipboard("hello")
        assert await device.get_clipboard() == "hello"
        await device.open_url("https://example.com")
        resources.location.set.assert_awaited_once_with(
            "abc", latitude=1.0, longitude=2.0
        )
        resources.location.reset.assert_awaited_once_with("abc")
        resources.language.set.assert_awaited_once_with("abc", locale="de-DE")
        resources.timezone.set.assert_awaited_once_with("abc", timezone="Europe/Berlin")
        resources.clipboard.set.assert_awaited_once_with("abc", text="hello")
        resources.deep_link.execute_deep_link.assert_awaited_once_with(
            "abc", deep_link="https://example.com"
        )

    asyncio.run(run())


def test_wrapper_and_pool_forward_open():
    async def run():
        inner = Device()
        inner.open_url = AsyncMock()
        wrapper = DeviceWrapper(inner)
        await wrapper.open_url("test://url")
        await DevicePool([wrapper]).open_url("test://url")
        assert inner.open_url.await_count == 2

    asyncio.run(run())


def test_android_open_quotes_and_rejects_invalid_settings():
    from mobilerun.core.devices.android import AndroidDevice

    async def run():
        device = AndroidDevice(serial="serial")
        device._ensure_connected = AsyncMock()
        shell = AsyncMock(return_value=SimpleNamespace(returncode=0, output=""))
        device._adb_device = cast(Any, SimpleNamespace(shell2=shell, serial="serial"))
        await device.open_url("https://example.com/?a=1&b='two'")
        assert shell.await_args is not None
        assert (
            shell.await_args.args[0]
            == "am start -a android.intent.action.VIEW -d 'https://example.com/?a=1&b='\"'\"'two'\"'\"''"
        )
        with pytest.raises(ValueError):
            await device.set_timezone("../bad")
        with pytest.raises(DeviceOperationNotSupported, match="physical Android"):
            await device.set_location(1, 2)

    asyncio.run(run())


def test_emulator_reset_explains_location_persists():
    from mobilerun.core.devices.android import AndroidDevice

    with pytest.raises(
        DeviceOperationNotSupported,
        match="emulators keep the last fixed location; set a new one instead",
    ):
        asyncio.run(AndroidDevice(serial="emulator-5554").reset_location())


def test_emulator_geo_fix_and_timezone():
    from unittest.mock import Mock

    from mobilerun.core.devices.android import AndroidDevice

    async def run():
        device = AndroidDevice(serial="emulator-5554")
        device._ensure_connected = AsyncMock()
        adb_output = Mock(return_value="OK")
        shell = AsyncMock(return_value=SimpleNamespace(returncode=0, output=""))
        device._adb_device = cast(
            Any,
            SimpleNamespace(
                serial="emulator-5554", adb_output=adb_output, shell2=shell
            ),
        )
        await device.set_location(52.52, 13.4)
        adb_output.assert_called_once_with("emu", "geo", "fix", "13.4", "52.52")
        await device.set_timezone("Europe/Berlin")
        shell.assert_awaited_once_with("cmd alarm set-timezone Europe/Berlin")
        with pytest.raises(ValueError):
            await device.set_timezone("Europe/Berlin;reboot")
        shell.return_value = SimpleNamespace(returncode=1, output="Permission denial")
        with pytest.raises(DeviceOperationNotSupported, match="Permission denial"):
            await device.set_timezone("Europe/Berlin")
        adb_output.return_value = "KO: denied"
        with pytest.raises(DeviceOperationNotSupported, match="rejected"):
            await device.set_location(0, 0)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("set_location", (1.0, 2.0)),
        ("reset_location", ()),
        ("set_timezone", ("Europe/Berlin",)),
        ("open_url", ("https://example.com",)),
        ("set_language", ("de-DE",)),
        ("set_clipboard", ("hello",)),
        ("get_clipboard", ()),
    ],
)
def test_wrapper_pool_delegate_each_environment_method(method, args):
    async def run():
        inner = Device()
        mock = AsyncMock(return_value="value" if method == "get_clipboard" else None)
        setattr(inner, method, mock)
        wrapper = DeviceWrapper(inner)
        assert await getattr(wrapper, method)(*args) == (
            "value" if method == "get_clipboard" else None
        )
        assert await getattr(DevicePool([wrapper]), method)(*args) == (
            ["value"] if method == "get_clipboard" else [None]
        )
        assert mock.await_count == 2
        mock.assert_awaited_with(*args)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("set_location", (1.0, 2.0)),
        ("reset_location", ()),
        ("set_timezone", ("Europe/Berlin",)),
        ("open_url", ("https://example.com",)),
    ],
)
@pytest.mark.parametrize("platform", ["ios", "remote"])
def test_unsupported_ios_remote(method, args, platform):
    from mobilerun.core.devices.ios import IOSDevice
    from mobilerun.core.devices.remote import RemoteDevice

    device = (
        IOSDevice("http://localhost:1")
        if platform == "ios"
        else RemoteDevice("http://localhost:1")
    )
    with pytest.raises(DeviceOperationNotSupported):
        asyncio.run(getattr(device, method)(*args))


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("reset_location", ()),
    ],
)
def test_unsupported_local_android(method, args):
    from mobilerun.core.devices.android import AndroidDevice

    with pytest.raises(DeviceOperationNotSupported):
        asyncio.run(getattr(AndroidDevice(serial="emulator-5554"), method)(*args))


@pytest.mark.parametrize("serial", ["emulator-5554", "physical-serial"])
@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("reset_location", ()),
    ],
)
def test_android_unsupported_environment_operations_on_both_kinds(serial, method, args):
    from mobilerun.core.devices.android import AndroidDevice

    with pytest.raises(DeviceOperationNotSupported):
        asyncio.run(getattr(AndroidDevice(serial=serial), method)(*args))
