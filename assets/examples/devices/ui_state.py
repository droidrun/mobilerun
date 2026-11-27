"""Read and interact with Android UI state through the device interface."""

import asyncio

from mobilerun.core.devices.android import AndroidDevice


async def main() -> None:
    async with AndroidDevice() as device:
        # Pass state_provider=custom_provider at construction to override state.
        state = await device.state.refresh()
        print(state.formatted_text)
        await device.tap_element(0)


if __name__ == "__main__":
    asyncio.run(main())
