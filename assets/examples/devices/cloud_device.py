"""Inspect a cloud device. Run: python assets/examples/devices/cloud_device.py DEVICE_ID. Requires cloud access."""

import argparse
import asyncio
from pathlib import Path

from mobilerun.core.devices import CloudDevice


async def run(device_id: str) -> None:
    # To override the logged-in credentials, pass api_key="..." explicitly.
    device = CloudDevice(device_id=device_id)
    await device._ensure_connected()
    Path("screenshot.png").write_bytes(await device.screenshot())
    print(await device.list_apps())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device_id")
    args = parser.parse_args()
    asyncio.run(run(args.device_id))


if __name__ == "__main__":
    main()
