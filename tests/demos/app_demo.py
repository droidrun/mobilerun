"""Install Counter and count on local and cloud devices.

Cloud tasks and AI can cost money. Run: python -m tests.demos.app_demo [--yes].
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pydantic import BaseModel

from mobilerun.config.settings import get_settings
from mobilerun.core.devices import AndroidDevice, CloudDevice, DevicePool
from mobilerun.core.operator.cloud import CloudTaskOperator

LOCAL_SERIAL = None
CLOUD_DEVICE_ID = "86f8d29f-9f59-4e42-ba02-1043c721774e"  # Tester 1
APK = get_settings().MOBILERUN_SMOKE_APK


class CounterValue(BaseModel):
    value: int


# ════════════════════════════════════════════════════════════════════════════
# COUNTER DEMO: install and count on both devices at the same time
# ════════════════════════════════════════════════════════════════════════════


async def install_and_open(pool: DevicePool) -> None:
    # Step 2: install the local APK on both devices.
    app = await pool.install_apk(APK)

    # Step 3: open Counter and let it settle.
    for device, installed in zip(pool.devices, app, strict=True):
        await device.start_app(installed.package)
    await pool.wait(2)


async def main(yes: bool = False) -> None:
    if not yes:
        try:
            answer = input("This demo uses paid cloud services. Continue? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled.")
            return
    print("── counter demo: start ──")

    # Step 1: connect a local agent and a cloud task device in a pool.
    device = AndroidDevice(LOCAL_SERIAL)
    cloud = CloudDevice(CLOUD_DEVICE_ID, operator=CloudTaskOperator())
    pool = DevicePool([device, cloud])
    try:
        await install_and_open(pool)

        # Step 4: count to ten and print each device's typed result.
        results = await pool.do(
            "Count to ten: tap the increment button 10 times",
            output=CounterValue,
        )
        for name, result in zip(("local", "cloud"), results, strict=True):
            print(name, "counter:", result.output.value if result.output else None)

        # Step 5: check the counter on both devices.

        try:
            expectations = await pool.expect("The counter shows 10")
        except Exception as e:
            print("expect failed:", e)
        else:
            for name, result in zip(("local", "cloud"), expectations, strict=True):
                print(name, "expect:", result.passed)

        # Step 6: return both devices to the home screen.
        await pool.press_button("home")
        print("── counter demo: done ──")
    finally:
        await pool.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("mobilerun.actions").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="Approve paid cloud usage")
    asyncio.run(main(yes=parser.parse_args().yes))
