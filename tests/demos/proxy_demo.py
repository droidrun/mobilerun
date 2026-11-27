"""Switch an existing cloud phone to another proxy, without creating a phone.

Run with ``python -m tests.demos.proxy_demo``. Selecting an existing
proxy is free; ``--country`` may purchase proxy capacity and requires approval.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import cast

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mobilerun.clients.cloud import devices, proxies  # noqa: E402
from mobilerun.core.devices import CloudDevice
from mobilerun.core.proxy import (
    CloudFixedProxy,
    CloudProxy,
    CloudRotatingProxy,
    Proxy,
)  # noqa: E402
from mobilerun.errors import ProxyNotActive  # noqa: E402
from mobilerun.models import Billing
from tests.demos.demo_billing import add_billing, billing_value, confirmation


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="tester1", help="Existing phone name")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--proxy-id", help="Existing active Connect proxy ID")
    selection.add_argument("--country", help="Country code; may incur charges")
    parser.add_argument(
        "--yes", action="store_true", help="Approve country proxy charges"
    )
    add_billing(parser)
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    billing = billing_value(args.billing)
    if args.country and billing != Billing.PAID:
        print("Country proxy selection requires --billing paid.", file=sys.stderr)
        return 1
    print(f"Billing: {billing.value}")
    if args.country and not args.yes:
        try:
            answer = input(confirmation(billing))
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Cancelled; no proxy was changed.", file=sys.stderr)
            return 1

    phones = await devices.list_devices(billing=Billing.NO_COMMERCE)
    matches = [item for item in phones if item.name == args.name]
    if len(matches) != 1:
        names = ", ".join(str(item.name) for item in phones) or "(none)"
        print(
            f"Phone {args.name!r} not found or not unique. Available: {names}",
            file=sys.stderr,
        )
        return 1

    phone = await CloudDevice(device_id=matches[0].id, billing=billing)
    try:
        status = await cast(CloudProxy, phone.proxy).status()
        old = status.name or "unknown"
        print(
            f"{args.name}: current proxy {old} ({'connected' if status.connected else 'disconnected'})"
        )
        items = []
        page = 1
        while True:
            result = await proxies.list_proxies(
                billing=Billing.NO_COMMERCE, page=page, page_size=100
            )
            items.extend(result.items)
            if not result.pagination.has_next:
                break
            page += 1
        for item in items:
            print(f"  {item.id}: {item.name} ({item.country}, {item.status})")
        current = (
            [item for item in items if item.name == status.name] if status.name else []
        )
        if len(current) == 1:
            old = f"{current[0].id} ({current[0].country})"
            print(f"Current proxy: {old}")
        if args.country:
            proxy: Proxy = CloudRotatingProxy(country=args.country)
            target = args.country
        else:
            if status.connected and len(current) != 1:
                print(
                    "Cannot identify the current proxy uniquely; refusing to switch.",
                    file=sys.stderr,
                )
                return 1
            options = [
                item
                for item in items
                if item.status == "active" and (not current or item.id != current[0].id)
            ]
            chosen = next(
                (
                    item
                    for item in options
                    if not args.proxy_id or item.id == args.proxy_id
                ),
                None,
            )
            if chosen is None:
                print(
                    "No active alternative proxy available for that selection.",
                    file=sys.stderr,
                )
                return 1
            proxy = CloudFixedProxy(id=chosen.id)
            target = f"{chosen.id} ({chosen.country})"
        try:
            await proxy.bind(phone, force=True)
            new_status = await cast(CloudProxy, phone.proxy).status()
        except ProxyNotActive as exc:
            print(f"Proxy switch failed: proxy not active ({exc})", file=sys.stderr)
            return 1
        new = new_status.name or target if args.country else target
        print(
            f"{args.name}: proxy {old} → {new} ({'connected' if new_status.connected else 'disconnected'})"
        )
        return 0
    finally:
        await phone.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
