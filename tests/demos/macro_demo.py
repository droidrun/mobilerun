"""Record and replay Android actions without AI or cloud charges.

Requires a local Android emulator with Portal installed. Run:
python -m tests.demos.macro_demo [--serial emulator-5554]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mobilerun.core.devices import AndroidDevice, RecordingDevice  # noqa: E402
from mobilerun.macro import MacroPlayer  # noqa: E402
from mobilerun.macro.models import MACRO_SCHEMA_VERSION  # noqa: E402


async def main(serial: str = "emulator-5554") -> None:
    device = AndroidDevice(serial)
    recorded = RecordingDevice(device)
    try:
        await recorded.start_app("com.android.settings")
        await recorded.tap(200, 200)
        await recorded.press_button("home")
        actions = list(recorded.recorded_actions)
        for step in actions:
            print(step)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "macro.json"
            path.write_text(
                json.dumps(
                    {"macro_schema_version": MACRO_SCHEMA_VERSION, "actions": actions}
                )
            )
            player = MacroPlayer(device)
            if not await player.replay_macro(player.load_macro_from_file(path)):
                raise RuntimeError("Macro replay failed")
    finally:
        await recorded.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial", default="emulator-5554", help="Local Android device serial"
    )
    asyncio.run(main(parser.parse_args().serial))
