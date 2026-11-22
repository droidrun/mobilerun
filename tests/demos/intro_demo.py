"""A simple introduction to Mobilerun devices."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pydantic import BaseModel  # noqa: E402

from mobilerun.clients.cloud.apps import upload_app  # noqa: E402
from mobilerun.config.settings import get_settings  # noqa: E402
from mobilerun.core.credentials import CloudCredentials, DictCredentials  # noqa: E402
from mobilerun.core.devices import (  # noqa: E402
    AndroidDevice,
    CloudDevice,
    Device,
    DevicePool,
    RecordingDevice,
)
from mobilerun.macro import MacroOperator, MacroRecorder  # noqa: E402
from mobilerun.macro.models import MACRO_SCHEMA_VERSION  # noqa: E402
from mobilerun.models import Billing  # noqa: E402
from mobilerun.sandbox import App, Context  # noqa: E402
from mobilerun.skills import list_skills  # noqa: E402
from tests.demos.demo_billing import confirmation  # noqa: E402

LOCAL_SERIAL = "emulator-5554"
CLOUD_DEVICE_ID = "86f8d29f-9f59-4e42-ba02-1043c721774e"  # Tester 1
AGENT = True  # do() and expect() use LLM time and cloud credits.
APK = get_settings().MOBILERUN_SMOKE_APK
NEW_CLOUD_DEVICE = False  # Creates a cloud phone, costs credits.
ANDROID_AVD = get_settings().MOBILERUN_ANDROID_DEVICE_TYPE


class AndroidVersion(BaseModel):
    version: str


def AgentOperator():
    """Load the optional operator only when the agent demo is enabled."""
    from mobilerun.agents.operator import AgentOperator as operator

    return operator()


# Define a sandbox task. Running it needs sandbox-api.
app = App("playground")


@app.task(device=CLOUD_DEVICE_ID)
def task(input: AndroidVersion, ctx: Context) -> AndroidVersion:
    return input


async def local_basics(device: Device) -> None:
    # Take a screenshot, count apps, and go home.
    Path("local-screenshot.png").write_bytes(await device.screenshot())
    print("local apps:", len(await device.list_apps()))
    await device.press_button("home")


async def cloud_basics(cloud: Device) -> None:
    # CloudDevice uses CloudTaskOperator by default.
    Path("cloud-screenshot.png").write_bytes(await cloud.screenshot())
    print("cloud apps:", len(await cloud.list_apps()))
    await cloud.press_button("home")


# ════════════════════════════════════════════════════════════════════════════
# MACRO DEMO: record one step, save it to a file, replay it without an LLM
# ════════════════════════════════════════════════════════════════════════════


async def macro_demo(device: Device) -> None:
    print("── macro demo: start ──")

    # Step 1: record actions while they happen on the device.
    recording = RecordingDevice(device)
    await recording.start_app("com.android.settings")
    await recording.wait(2)
    await recording.swipe(500, 1600, 500, 500, 500)
    await recording.wait(1)
    await recording.press_button("back")
    recorder = MacroRecorder()
    # UI state can differ between runs, so leave these demo actions unguarded.
    for action in recording.recorded_actions:
        recorder.record_action(action)["pre_state"] = None

    with TemporaryDirectory() as directory:
        # Step 2: save the recorded actions as a macro file.
        path = Path(directory) / "macro.json"
        path.write_text(
            json.dumps(
                {
                    "macro_schema_version": MACRO_SCHEMA_VERSION,
                    "actions": recorder.actions,
                }
            )
        )

        # Step 3: return home, then replay with a one-call operator.
        await device.press_button("home")
        result = await device.do("Replay", operator=MacroOperator(path))
        print("macro replay:", result.success)

    print("── macro demo: done ──")


# ════════════════════════════════════════════════════════════════════════════
# CREDENTIALS DEMO: type a local credential without exposing it to the model
# ════════════════════════════════════════════════════════════════════════════


async def credentials_demo(device: Device) -> None:
    print("── credentials demo: start ──")

    # Step 1: attach a local credential provider.
    vault = DictCredentials({"demo_password": "not-a-real-secret"})
    await vault.bind(device)

    # Step 2: open Settings and focus its search field.
    await device.start_app("com.android.settings")
    await device.wait(2)
    await device.state.refresh()
    search = next(
        item for item in device.state.current.elements if "search" in str(item).lower()
    )
    await device.tap_element(search["index"])

    # Step 3: type directly; the value is never logged or sent to the model.
    await vault.type_field(None, "password", name="demo_password")
    await device.press_button("home")
    print("── credentials demo: done ──")


# ════════════════════════════════════════════════════════════════════════════
# VAULT DEMO: save, list, and delete a throwaway cloud credential
# ════════════════════════════════════════════════════════════════════════════


async def vault_demo() -> None:
    print("── vault demo: start ──")

    # Step 1: cloud tasks only accept CloudCredentials (vault), not local providers.
    vault = CloudCredentials()
    package = "ai.mobilerun.demo"
    name = f"demo-{uuid4().hex}"
    await vault.save(package, name, {"password": "not-a-real-secret"})
    try:
        # Step 2: list names without printing values.
        print("vault names:", [item["name"] for item in await vault.list(package)])
    finally:
        # Step 3: delete the throwaway credential even if listing fails.
        await vault.delete(package, name)
    print("── vault demo: done ──")


# ════════════════════════════════════════════════════════════════════════════
# APP INSTALL DEMO: install locally and on the cloud phone
# ════════════════════════════════════════════════════════════════════════════


async def local_app_install_demo(device: Device) -> None:
    print("── local app install demo: start ──")
    # Step 1: keep the local Counter app installed for later use.
    if Path(APK).is_file():
        await device.install_apk(APK, reinstall=True)
    else:
        print("Set MOBILERUN_SMOKE_APK to an existing APK to run the install demo.")
    print("── local app install demo: done ──")


async def cloud_app_install_demo(cloud: Device) -> None:
    print("── cloud app install demo: start ──")
    if Path(APK).is_file():
        # Step 1: upload or reuse the APK in the cloud app library.
        app = await upload_app(APK, billing=get_settings().MOBILERUN_BILLING)
        print("app package:", app.package, "version:", app.version)
        # Step 2: install, then remove it from the cloud phone only.
        await cloud.install_app(app)
        await cloud.uninstall_app(app.package)
    else:
        print("Set MOBILERUN_SMOKE_APK to an existing APK to run the install demo.")
    print("── cloud app install demo: done ──")


# ════════════════════════════════════════════════════════════════════════════
# NEW CLOUD DEVICE DEMO: create a temporary paid cloud phone
# ════════════════════════════════════════════════════════════════════════════


async def new_cloud_device_demo() -> None:
    print("── new cloud device demo: start ──")
    if NEW_CLOUD_DEVICE:
        # Step 1: the phone is automatically terminated when the context ends.
        async with CloudDevice(billing="paid") as fresh:
            print("new cloud device:", fresh.device_id)
            Path("fresh-cloud-screenshot.png").write_bytes(await fresh.screenshot())
    else:
        print("new cloud device: skip (NEW_CLOUD_DEVICE is False)")
    print("── new cloud device demo: done ──")


# ════════════════════════════════════════════════════════════════════════════
# EMULATOR DEMO: reuse or start the configured Android virtual device
# ════════════════════════════════════════════════════════════════════════════


async def emulator_demo() -> None:
    print("── emulator demo: start ──")
    if ANDROID_AVD:
        # Step 1: a running AVD is reused; otherwise the configured AVD starts.
        emulator = await AndroidDevice(avd=ANDROID_AVD)
        try:
            Path("emulator-screenshot.png").write_bytes(await emulator.screenshot())
        finally:
            await emulator.close()
    else:
        print("emulator: skip (set MOBILERUN_ANDROID_DEVICE_TYPE)")
    print("── emulator demo: done ──")


async def agent_demo(cloud: Device) -> None:
    # Ask the cloud agent to act, check the result, and return typed output.
    if AGENT:
        result = await cloud.do("Open the Battery settings")
        print("do:", result.success)
        result = await cloud.expect("The Battery settings are open")
        print("expect:", result.passed)
        result = await cloud.do("Find the Android version", output=AndroidVersion)
        if result.success and result.output:
            print("Android version:", result.output.version)
        else:
            print(f"typed output: failed ({result.answer})")


# ════════════════════════════════════════════════════════════════════════════
# POOL DEMO: one call runs on both devices at the same time
# ════════════════════════════════════════════════════════════════════════════


async def pool_demo(device: Device, cloud: Device) -> None:
    print("── pool demo: start ──")

    # Step 1: group both devices into a pool.
    pool = DevicePool([device, cloud])

    # Step 2: every pool call runs on all devices and returns one result each.
    images = await pool.screenshot()
    print("pool screenshots:", len(images))
    await pool.press_button("home")

    # Step 3 (AGENT only): each device uses its own operator.
    if AGENT:
        print("pool do:", await pool.do("Open the Settings app"))
        print("pool expect:", await pool.expect("The Settings app is open"))

    print("── pool demo: done ──")


def sandbox_and_skills() -> None:
    # Show the sandbox task without running it.
    print("sandbox app:", app.name, "task:", task.__name__)

    # List the bundled skills.
    print("skills:", [skill.name for skill in list_skills()])


async def local_demos(device: Device) -> None:
    await local_basics(device)
    await macro_demo(device)
    await credentials_demo(device)
    await local_app_install_demo(device)


async def cloud_demos(cloud: Device) -> None:
    await cloud_basics(cloud)
    await agent_demo(cloud)
    await vault_demo()
    await cloud_app_install_demo(cloud)


async def main() -> None:
    # Connect to both devices.
    if AGENT:
        device = AndroidDevice(LOCAL_SERIAL, operator=AgentOperator())
    else:
        device = AndroidDevice(LOCAL_SERIAL)
    cloud = CloudDevice(CLOUD_DEVICE_ID)
    await device._ensure_connected()
    await cloud._ensure_connected()
    sandbox_and_skills()

    # Run both devices at once, keeping each device's demos in order.
    await asyncio.gather(local_demos(device), cloud_demos(cloud))

    # The pool uses both devices, so run it after their individual demos.
    await pool_demo(device, cloud)
    await new_cloud_device_demo()
    await emulator_demo()


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="Approve paid usage")
    args = parser.parse_args(argv)
    if not args.yes:
        costs = "Cloud task and LLM use can cost money"
        if NEW_CLOUD_DEVICE:
            costs += "; a new cloud phone can also cost money"
        try:
            answer = input(f"{costs}. {confirmation(Billing.PAID)}")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled.")
            return
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("mobilerun.actions").setLevel(logging.INFO)
    asyncio.run(main())


if __name__ == "__main__":
    cli()
