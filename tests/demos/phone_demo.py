"""Rent a phone number and wait for an SMS code.

The number may cost money; --bind also creates a cloud phone. Run:
python -m tests.demos.phone_demo [--country US] [--bind] [--yes]
"""

import argparse
import asyncio
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mobilerun.core.devices import CloudDevice  # noqa: E402
from mobilerun.core.phone_number import CloudPhoneNumber  # noqa: E402
from mobilerun.errors import PaymentRequired, PurchaseInProgress  # noqa: E402
from tests.demos.demo_billing import (  # noqa: E402
    add_billing,
    billing_value,
    check_phone_plan,
    confirmation,
)


async def main(args: argparse.Namespace) -> None:
    billing = billing_value(getattr(args, "billing", None))
    plan = getattr(args, "phone_plan", getattr(args, "plan", None))
    number_plan = getattr(args, "number_plan", None)
    if args.bind:
        check_phone_plan(billing, plan)
    print(
        f"Billing: {billing.value}; number plan: {number_plan or 'auto'}"
        + (
            f"; phone plan: {plan or ('subscription' if billing == 'included' else 'auto')}"
            if args.bind
            else ""
        )
    )
    if not args.yes:
        try:
            answer = input(confirmation(billing))
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled.")
            return
    try:
        number = await CloudPhoneNumber(
            country=args.country or "US",
            billing=billing,
            **({"plan": number_plan} if number_plan is not None else {}),
            checkout="print",
            keep=args.keep,
        )
    except (PaymentRequired, PurchaseInProgress) as exc:
        print(str(exc))
        if exc.checkout:
            print(f"Checkout: {exc.checkout.url}")
        return
    try:
        # Creation can return before the actual phone number is provisioned.
        await number.wait_ready()
        if not number.phone_number:
            raise RuntimeError("The active number has no phone number")
        if args.bind:
            async with CloudDevice(billing=billing, plan=plan) as phone:
                try:
                    await number.bind(phone)
                    print(f"phone.number is number: {phone.number is number}")
                    print(f"phone.sms is number: {phone.sms is number}")
                    print(f"Send an SMS with a code to {number.phone_number} …")
                    code = await number.wait_for_code(timeout=args.timeout)
                    print(f"Received code: {code}")
                finally:
                    await number.unbind()
        else:
            print(f"Send an SMS with a code to {number.phone_number} …")
            code = await number.wait_for_code(timeout=args.timeout)
            print(f"Received code: {code}")
    finally:
        if getattr(number, "resumed_existing", False) is True:
            print(f"Preserved existing number {number.id}; it may keep costing.")
        elif args.keep:
            print(f"Kept number {number.id}; it may keep costing.")
        else:
            await number.release()
            print(f"Released number {number.id}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", help="Country code (default: US)")
    parser.add_argument("--yes", action="store_true", help="Approve paid resources")
    parser.add_argument(
        "--keep", action="store_true", help="Keep the number after the demo"
    )
    parser.add_argument(
        "--bind", action="store_true", help="Create a cloud phone and bind the number"
    )
    parser.add_argument(
        "--timeout", type=float, default=180, help="Code wait timeout in seconds"
    )
    add_billing(parser)
    parser.add_argument(
        "--phone-plan",
        "--plan",
        choices=("auto", "minute", "subscription"),
        help="Cloud phone plan (requires --bind)",
    )
    parser.add_argument(
        "--number-plan",
        choices=("auto", "included", "rent"),
        help="Number plan (default: auto)",
    )
    args = parser.parse_args()
    try:
        if args.bind:
            check_phone_plan(billing_value(args.billing), args.phone_plan)
        elif args.phone_plan:
            parser.error("--phone-plan requires --bind")
    except ValueError as exc:
        parser.error(str(exc))
    return args


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
