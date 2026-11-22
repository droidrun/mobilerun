"""Inspect iOS. Run: python assets/examples/devices/ios_device.py PORTAL_URL. Requires a reachable iOS Portal.

For token handling and Portal detection, use IOSDevice from mobilerun.core.devices.
"""

import argparse
import asyncio
from pathlib import Path

from mobilerun.core.devices import IOSDevice


async def run(url: str) -> None:
    device = IOSDevice(url=url)
    await device._ensure_connected()
    print(device.platform, sorted(device.capabilities))
    Path("screenshot.png").write_bytes(await device.screenshot())
    print("UI tree keys:", list(await device._get_ui_tree()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    args = parser.parse_args()
    asyncio.run(run(args.url))


if __name__ == "__main__":
    main()
