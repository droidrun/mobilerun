"""Run tasks on a temporary cloud phone.

The phone, proxy and cloud tasks cost money. Run: python -m tests.demos.task_demo [--yes].
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pydantic import BaseModel

from mobilerun.core.devices import CloudDevice
from mobilerun.core.proxy import CloudRotatingProxy
from mobilerun.models import Billing
from tests.demos.demo_billing import (
    add_billing,
    billing_value,
    check_phone_plan,
    confirmation,
)


class AndroidVersion(BaseModel):
    version: str


async def main(
    yes: bool = False, billing: Billing | None = None, plan: str | None = None
) -> None:
    billing = billing_value(billing)
    check_phone_plan(billing, plan)
    print(
        f"Billing: {billing.value}; phone plan: {plan or ('subscription' if billing == 'included' else 'auto')}"
    )
    if not yes:
        try:
            answer = input(confirmation(billing))
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled.")
            return
    print("── cloud phone demo: start ──")

    # Step 1: Create a temporary phone with an explicit US proxy.
    # The phone is terminated automatically.
    # Pass country="DE" to select the device's country.
    proxy = CloudRotatingProxy(country="US")
    async with CloudDevice(
        name="cloud-phone-demo",
        billing=billing,
        plan=plan,
        proxy=proxy,
    ) as phone:
        print("phone:", phone.device_id)
        status = await proxy.status()
        print("proxy connected:", status.connected)
        print("proxy country requested: US")

        # Step 2: Use the default cloud task operator to open Settings.
        # Pass operator=AgentOperator() to use your local agent instead.
        result = await phone.do("Open the Settings app")
        print("open task:", result.task_id)
        check = await phone.expect("The Settings app is open")
        print("expect task:", check.task.task_id)

        # Step 3: Ask for typed output, then run a task with a timeout.
        result = await phone.do("Find the Android version", output=AndroidVersion)
        print("version task:", result.task_id)
        if result.output is None:
            raise RuntimeError("Android version task returned no output")
        print("Android version:", result.output.version)
        result = await phone.do("Return to the main Settings screen", timeout=120)
        print("timed task:", result.task_id)

        # Step 4: Return home before automatic termination.
        await phone.press_button("home")
    print("── cloud phone demo: done ──")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("mobilerun.actions").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="Approve paid cloud usage")
    add_billing(parser, plan=True)
    args = parser.parse_args()
    try:
        check_phone_plan(billing_value(args.billing), args.plan)
    except ValueError as exc:
        parser.error(str(exc))
    asyncio.run(main(yes=args.yes, billing=billing_value(args.billing), plan=args.plan))
