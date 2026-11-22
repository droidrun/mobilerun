"""Extend a device. Run: python assets/examples/devices/custom_device.py [serial]. Requires Android and Portal."""

import argparse
import asyncio
from pathlib import Path

from mobilerun.config.settings import get_settings
from mobilerun.core.devices import AndroidDevice


class MyAndroidDevice(AndroidDevice):
    async def screenshot_to(self, path: str) -> None:
        Path(path).write_bytes(await self.screenshot())


async def run(serial: str | None = None, device: MyAndroidDevice | None = None) -> None:
    device = device or MyAndroidDevice(serial=serial or get_settings().android_serial)
    await device._ensure_connected()
    await MyAndroidDevice.screenshot_to(device, "screenshot.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", nargs="?")
    args = parser.parse_args()
    asyncio.run(run(args.serial))


if __name__ == "__main__":
    main()
