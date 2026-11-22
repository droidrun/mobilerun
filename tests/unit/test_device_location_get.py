import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner
from mobilerun_sdk.types.shared.location import Location

from mobilerun.cli.devices.base import selection
from mobilerun.cli.devices.group import device
from mobilerun.cli.devices.settings import location as location_command
from mobilerun.clients.cloud import devices
from mobilerun.core.devices.android import AndroidDevice
from mobilerun.core.devices.base import Device
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.devices.location import Location as DeviceLocation
from mobilerun.core.devices.pool import DevicePool
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.models.billing import Billing


def test_cloud_location_helper_and_device():
    async def run():
        value = Location(latitude=52.52, longitude=13.405)
        get = AsyncMock(return_value=value)
        client = SimpleNamespace(
            devices=SimpleNamespace(location=SimpleNamespace(get=get))
        )
        assert (
            await devices.get_location(
                "abc", billing=Billing.NO_COMMERCE, client=client
            )
            == value
        )
        phone = CloudDevice("abc", api_key="test")
        phone._client = client
        assert await phone.get_location() == DeviceLocation(
            latitude=52.52, longitude=13.405
        )
        assert get.await_count == 2
        get.assert_awaited_with("abc")

    asyncio.run(run())


def test_unsupported_and_wrapped_pool():
    async def run():
        with pytest.raises(DeviceOperationNotSupported, match="get_location"):
            await AndroidDevice(serial="emulator-5554").get_location()
        inner = Device()
        inner.get_location = AsyncMock(return_value=Location(latitude=1, longitude=2))
        assert (await DevicePool([DeviceWrapper(inner)]).get_location())[
            0
        ].latitude == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "value,args,expected",
    [
        (Location(latitude=52.52, longitude=13.405), [], "Location: 52.5200, 13.4050"),
        (None, [], "Location: Not available"),
        (
            Location(latitude=52.52, longitude=13.405),
            ["--json"],
            {"latitude": 52.52, "longitude": 13.405},
        ),
        (None, ["--json"], None),
    ],
)
def test_cli_location_get(monkeypatch, value, args, expected):
    phone = SimpleNamespace(
        get_location=AsyncMock(return_value=value), close=AsyncMock()
    )
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=phone))
    result = CliRunner().invoke(
        device, ["settings", "location", "get", "--device", "abc", *args]
    )
    assert result.exit_code == 0, result.output
    assert (json.loads(result.output) if args else result.output.strip()) == expected
    phone.close.assert_awaited_once()


def test_location_menu_lists_get():
    result = CliRunner().invoke(device, ["settings", "location", "--help"])
    assert "get" in result.output


def test_interactive_location_menu_lists_get(monkeypatch):
    from mobilerun.cli.base import menu

    seen = []
    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    def choose(label, choices):
        seen.extend(choice.value for choice in choices)
        return None

    monkeypatch.setattr(menu, "select_prompt", choose)
    CliRunner().invoke(location_command, [])
    assert {"get", "set", "reset"} <= set(seen)
