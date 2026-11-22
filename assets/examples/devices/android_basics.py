"""Control Android basics. Run: python assets/examples/devices/android_basics.py [serial]. Requires ADB and Portal."""

import argparse
import asyncio
from pathlib import Path

from mobilerun.config.settings import get_settings
from mobilerun.core.devices import AndroidDevice


async def run(serial: str | None = None, device: AndroidDevice | None = None) -> None:
    device = device or AndroidDevice(serial=serial or get_settings().android_serial)
    await device._ensure_connected()
    print(
        device.platform, sorted(device.capabilities), sorted(device.supported_buttons)
    )
    Path("screenshot.png").write_bytes(await device.screenshot())
    await device.start_app("com.android.settings")
    # Coordinates are examples: adjust them for your device and focused field.
    await device.tap(200, 200)
    await device.swipe(200, 600, 200, 300, 500)
    await device.input_text("Hello from Mobilerun")
    await device.press_button("back")
    await device.press_button("home")
    tree = await device._get_ui_tree()
    print("UI tree keys:", list(tree))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", nargs="?")
    args = parser.parse_args()
    asyncio.run(run(args.serial))


if __name__ == "__main__":
    main()
