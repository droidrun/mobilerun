"""Create a mailbox and wait for an emailed code.

The public mailbox API exposes code polling, not sending mail or message
contents (sender, subject, and body). Send a message containing a code from
another email account. A mailbox costs money. Run: python -m tests.demos.mailbox_demo.
Authenticate with MOBILERUN_API_KEY or mobilerun account login.
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mobilerun.core.mailbox import CloudMailbox
from mobilerun.errors import PaymentRequired
from tests.demos.demo_billing import add_billing, billing_value, confirmation


async def main(args: argparse.Namespace) -> None:
    billing = billing_value(getattr(args, "billing", None))
    print(f"Billing: {billing.value}")
    if not args.yes:
        try:
            answer = input(confirmation(billing))
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            return
    try:
        mailbox = await CloudMailbox(
            billing=billing,
            keep=args.keep,
            checkout="print",
            payment_timeout=args.payment_timeout,
        )
    except PaymentRequired as exc:
        print(str(exc))
        print(f"Checkout: {exc.checkout.url}")
        return
    try:
        # Capture the start before displaying the address so fast messages count.
        after = datetime.now(timezone.utc)
        print(f"Mailbox ID: {mailbox.id}")
        print(f"Send an email containing a code to {mailbox.address} now…")
        code = await mailbox.wait_for_code(timeout=args.timeout, after=after)
        print(f"Received code: {code}")
    finally:
        if not args.keep:
            await mailbox.release()
            print(f"Released mailbox {mailbox.id}.")
        else:
            print(f"Kept mailbox {mailbox.id}; it may continue to incur charges.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes", action="store_true", help="skip paid resource confirmation"
    )
    parser.add_argument(
        "--timeout", type=float, default=180, help="code wait timeout in seconds"
    )
    parser.add_argument(
        "--payment-timeout",
        type=float,
        default=600,
        help="payment wait timeout in seconds",
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the mailbox after the demo"
    )
    add_billing(parser)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
