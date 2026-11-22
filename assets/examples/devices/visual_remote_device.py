"""Inspect a remote device. Run: python assets/examples/devices/visual_remote_device.py [URL]. Requires a visual remote server."""

import argparse
import asyncio
from pathlib import Path

from mobilerun.core.devices import RemoteDevice


async def run(url: str = "http://localhost:8090", device_id: str = "auto") -> None:
    device = RemoteDevice(url=url, device_id=device_id)
    await device._ensure_connected()
    print(
        device.platform, sorted(device.capabilities), sorted(device.supported_buttons)
    )
    Path("screenshot.png").write_bytes(await device.screenshot())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", default="http://localhost:8090")
    parser.add_argument("--device-id", default="auto")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.device_id))


if __name__ == "__main__":
    main()
