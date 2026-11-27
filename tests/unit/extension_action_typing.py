"""Static Pyright fixture: injected Device is absent from caller signatures."""

from typing import assert_type

from mobilerun.core.devices import Device
from mobilerun.core.extensions import Extension, action, device_action


class TypedActions(Extension):
    @action
    async def list(self, package: str) -> str:
        """Read standalone metadata."""
        return package

    @device_action
    async def type_field(self, device: Device, package: str, field: str) -> None:
        """Use the nonoptional injected device."""
        assert_type(device, Device)
        await device.input_text(package + field)

    @device_action(public=False)
    async def configured(self, device: Device, value: int = 1) -> int:
        """Exercise configured decorator inference."""
        assert_type(device, Device)
        return value


async def check_public_signatures(vault: TypedActions) -> None:
    assert_type(await vault.list("pkg"), str)
    assert_type(await vault.type_field("pkg", "username"), None)
    assert_type(await vault.configured(), int)
    assert_type(await vault.configured(2), int)
