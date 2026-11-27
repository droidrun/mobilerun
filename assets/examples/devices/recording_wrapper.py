"""Compose driver decorators. Run: python assets/examples/devices/recording_wrapper.py [serial]. Requires Android and Portal."""

import argparse
import asyncio

from mobilerun.config.settings import get_settings
from mobilerun.core.devices import AndroidDevice, RecordingDevice, StealthDevice


async def run(serial: str | None = None, device: AndroidDevice | None = None) -> None:
    device = device or AndroidDevice(serial=serial or get_settings().android_serial)
    wrapped = RecordingDevice(StealthDevice(device))
    await wrapped._ensure_connected()
    await wrapped.tap(200, 200)
    print(wrapped.actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", nargs="?")
    args = parser.parse_args()
    asyncio.run(run(args.serial))


if __name__ == "__main__":
    main()
