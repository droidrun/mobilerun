"""Record and replay Android actions without AI or cloud charges.

Requires a local Android emulator with Portal installed. Run:
python -m tests.demos.macro_demo [--serial emulator-5554]
"""

import argparse
import asyncio
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mobilerun.core.devices import AndroidDevice, RecordingDevice  # noqa: E402
from mobilerun.macro import MacroPlayer, build_macro, load_macro  # noqa: E402


async def main(serial: str = "emulator-5554") -> None:
    device = AndroidDevice(serial)
    async with device:
        await device.press_button("home")
        recorded = RecordingDevice(device, capture="actions")
        await recorded.start_app("com.android.settings")
        await recorded.tap(200, 200)
        await recorded.press_button("home")
        recorded.finish("succeeded")
        assert recorded.run_dir is not None
        path = build_macro(recorded.run_dir / "steps.jsonl", "settings-demo")
        print("macro version:", path)
        await MacroPlayer(device).replay_macro(load_macro(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial", default="emulator-5554", help="Local Android device serial"
    )
    asyncio.run(main(parser.parse_args().serial))
