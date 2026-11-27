"""No-commerce choices for interactive command parameters."""

import asyncio

import click
from async_adbutils import adb

from mobilerun.cli.base.prompts import SelectChoice
from mobilerun.clients.cloud.devices import list_devices
from mobilerun.clients.cloud.mailboxes import list_mailboxes
from mobilerun.clients.cloud.numbers import list_numbers
from mobilerun.clients.cloud.proxies import list_proxies
from mobilerun.models.billing import Billing


def connected_devices(ctx: click.Context) -> list[SelectChoice]:
    return [SelectChoice(item.serial, item.serial) for item in asyncio.run(adb.list())]


def cloud_devices(ctx: click.Context) -> list[SelectChoice]:
    items = asyncio.run(
        list_devices(billing=Billing.NO_COMMERCE, base_url=ctx.params.get("base_url"))
    )
    items.sort(key=lambda item: str(item.state).lower() not in {"running", "ready"})
    return [
        SelectChoice(
            str(item.id),
            str(item.name or item.id),
            " · ".join(
                str(value)
                for value in (item.state, getattr(item, "country", None))
                if value
            ),
        )
        for item in items
    ]


def mailboxes(ctx: click.Context) -> list[SelectChoice]:
    items = asyncio.run(list_mailboxes(billing=Billing.NO_COMMERCE))
    items.sort(key=lambda item: getattr(item, "status", None) != "active")
    return [
        SelectChoice(
            str(item.id), str(item.address or item.id), getattr(item, "status", None)
        )
        for item in items
    ]


def numbers(ctx: click.Context) -> list[SelectChoice]:
    items = asyncio.run(list_numbers(billing=Billing.NO_COMMERCE))
    items.sort(
        key=lambda item: getattr(item, "state", None)
        not in {"active", "cancel_scheduled"}
    )
    return [
        SelectChoice(
            str(item.id),
            str(item.phone_number or item.id),
            getattr(item, "state", None),
        )
        for item in items
    ]


def proxies(ctx: click.Context) -> list[SelectChoice]:
    async def fetch() -> list[SelectChoice]:
        page = 1
        items = []
        while True:
            response = await list_proxies(
                billing=Billing.NO_COMMERCE, page=page, page_size=100
            )
            items.extend(response.items)
            if not response.pagination.has_next:
                items.sort(key=lambda item: getattr(item, "status", None) != "active")
                return [
                    SelectChoice(
                        str(item.id),
                        str(item.id),
                        " · ".join(
                            str(value)
                            for value in (
                                item.country,
                                getattr(item, "status", None),
                                getattr(item, "type", None),
                            )
                            if value
                        ),
                    )
                    for item in items
                ]
            page += 1

    return asyncio.run(fetch())
