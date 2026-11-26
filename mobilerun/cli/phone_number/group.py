"""Explicit communications commands; only these commands display codes."""

import asyncio
import json

import click
from rich.text import Text

from mobilerun.cli.base import render
from mobilerun.cli.base.billing import (
    billing_option,
    invoke_with_billing,
)
from mobilerun.cli.base.checkout import complete_checkout
from mobilerun.cli.base.choices import cloud_devices, numbers
from mobilerun.cli.base.cleanup import cleanup
from mobilerun.cli.base.menu import MenuGroup, PromptingArgument, is_interactive
from mobilerun.cli.base.prompts import (
    SelectChoice,
)
from mobilerun.cli.base.prompts import (
    select_prompt_async as select_prompt,
)
from mobilerun.cli.base.run import coro
from mobilerun.clients.cloud import devices
from mobilerun.clients.cloud import numbers as cloud_numbers
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.clients.cloud.numbers import list_numbers
from mobilerun.config.settings import get_settings
from mobilerun.core.phone_number import CloudPhoneNumber as PhoneNumber
from mobilerun.errors import (
    CloudAuthenticationError,
    CloudPermissionDenied,
    ExtensionActionError,
)
from mobilerun.errors.base import MobilerunError
from mobilerun.models.billing import Billing


@click.group(cls=MenuGroup)
def number() -> None:
    """Manage cloud phone numbers."""


async def _included_counts(countries: list[str], billing: str) -> list[int | None]:
    if not countries:
        return []
    semaphore = asyncio.Semaphore(
        get_settings().MOBILERUN_CLOUD_NUMBER_CAPACITY_CONCURRENCY
    )
    async with cloud_client(billing=Billing(billing)) as client:

        async def lookup(country: str) -> int:
            async with semaphore:
                return await PhoneNumber.included_available(
                    country, billing=billing, client=client
                )

        results = await asyncio.gather(
            *(lookup(country) for country in countries), return_exceptions=True
        )
    for result in results:
        if isinstance(result, (CloudAuthenticationError, CloudPermissionDenied)):
            raise result
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            raise result
    failed = [
        country
        for country, result in zip(countries, results, strict=True)
        if isinstance(result, Exception)
    ]
    if failed:
        click.echo(
            f"Warning: included count unavailable for {', '.join(failed)}.", err=True
        )
    return [result if isinstance(result, int) else None for result in results]


@number.command("messages", help="List SMS message history.")
@click.argument("number_id", cls=PromptingArgument, choices=numbers, metavar="NUMBER")
@click.option("--json", "as_json", is_flag=True)
@billing_option
@coro
async def number_messages(number_id: str, as_json: bool, billing: str) -> None:
    items = await cloud_numbers.messages(number_id, billing=Billing(billing))
    entries = [item.model_dump(mode="json") for item in items]
    if as_json:
        click.echo(json.dumps(entries, ensure_ascii=False))
    elif not entries:
        render.empty("messages")
    else:
        render.heading(len(entries), "message")
        render.table(
            [
                [
                    entry.get("peer_number")
                    or entry.get("detected_sender")
                    or entry.get("id"),
                    entry.get("body") or "",
                    render.secondary(entry.get("direction")),
                    render.status(entry.get("status")),
                    render.relative_time(entry.get("occurred_at")),
                ]
                for entry in entries
            ]
        )


@number.command("countries", help="List supported number countries.")
@click.option("--json", "as_json", is_flag=True)
@billing_option
@coro
async def number_countries(as_json: bool, billing: str) -> None:
    result = await cloud_numbers.countries(billing=Billing(billing))
    entries = [item.model_dump(mode="json") for item in result.data.items]
    if as_json:
        click.echo(json.dumps(entries, ensure_ascii=False))
    elif not entries:
        render.empty("countries")
    else:
        render.heading(len(entries), "country")
        render.table(
            [
                [
                    entry["name"],
                    render.secondary(entry["country"]),
                    render.status("available" if entry["in_stock"] else "out of stock"),
                ]
                for entry in entries
            ]
        )


@number.command("send", help="Send an SMS (may incur charges).")
@click.argument("number_id", cls=PromptingArgument, choices=numbers, metavar="NUMBER")
@click.argument("to")
@click.argument("text")
@click.option("--yes", "-y", is_flag=True)
@billing_option
@coro
async def number_send(
    number_id: str, to: str, text: str, yes: bool, billing: str
) -> None:
    await invoke_with_billing(
        lambda mode: cloud_numbers.send(number_id, to, text, billing=mode),
        Billing(billing),
        yes=yes,
    )
    render.success(f"Sent SMS from {number_id} to {to}")


@number.command("status", help="Show a number's lifecycle state and binding.")
@click.argument("number_id", cls=PromptingArgument, choices=numbers, metavar="NUMBER")
@click.option("--json", "as_json", is_flag=True)
@billing_option
@coro
async def number_status(number_id: str, as_json: bool, billing: str) -> None:
    data = (await cloud_numbers.retrieve(number_id, billing=Billing(billing))).data
    entry = {
        "id": data.id,
        "status": data.state,
        "phone_number": data.phone_number,
        "device_id": data.device_binding.device_id if data.device_binding else None,
    }
    if as_json:
        click.echo(json.dumps(entry))
    else:
        render.table(
            [
                [
                    entry["phone_number"] or entry["id"],
                    render.status(entry["status"]),
                    render.secondary(f"device: {entry['device_id'] or 'none'}"),
                    render.secondary(entry["id"]) if entry["phone_number"] else "",
                ]
            ]
        )


@number.command("list", help="List cloud phone numbers.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@billing_option
@coro
async def number_list(billing: str, as_json: bool) -> None:
    items = await list_numbers(billing=Billing(billing))
    entries = [
        {
            "id": item.id,
            "phone_number": item.phone_number,
            "country_code": item.country_code,
            "state": item.state,
        }
        for item in items
    ]
    if as_json:
        click.echo(json.dumps(entries, ensure_ascii=False))
    elif not entries:
        render.empty("phone numbers", "Create one: mobilerun number create")
    else:
        render.heading(len(entries), "phone number")
        render.table(
            [
                [
                    Text(str(entry["phone_number"] or entry["id"])),
                    render.secondary(entry["country_code"]),
                    render.status(entry["state"]),
                    render.secondary(entry["id"]) if entry["phone_number"] else "",
                ]
                for entry in entries
            ]
        )


@number.command("create", help="Create a cloud phone number.")
@click.option("--json", "as_json", is_flag=True)
@click.option(
    "--plan",
    type=click.Choice(["auto", "included", "rent"]),
    default="auto",
    show_default=True,
    help="auto = included first, then rent if billing is paid; included = never rent; rent = always rent (requires paid billing).",
)
@click.option("--yes", "-y", is_flag=True, help="Confirm without prompting.")
@click.option(
    "--payment-timeout",
    type=click.FloatRange(min=0, min_open=True),
    default=600,
    help="Payment wait timeout in seconds.",
)
@click.option("--country", default=None, help="Country code for the number.")
@billing_option
@coro
async def number_create(
    country: str | None,
    billing: str,
    yes: bool,
    payment_timeout: float,
    plan: str,
    as_json: bool,
) -> None:
    if not country and is_interactive():
        available = await cloud_numbers.countries(billing=Billing(billing))
        in_stock = [item for item in available.data.items if item.in_stock]
        remaining = await _included_counts([item.country for item in in_stock], billing)
        choices = [
            SelectChoice(
                item.country,
                f"{item.name} ({item.country}) · "
                + (
                    "included count unavailable"
                    if count is None
                    else f"{count} included" + (" left" if count else "")
                ),
            )
            for item, count in zip(in_stock, remaining, strict=True)
        ]
        if not choices:
            raise click.ClickException(
                "No phone number countries are currently in stock."
            )
        country = await select_prompt("Choose a number country", choices)
        if country is None:
            raise click.Abort()
    item, mode = await invoke_with_billing(
        lambda attempt: complete_checkout(
            PhoneNumber(
                country=country,
                billing=attempt,
                plan=plan,
                keep=True,
                checkout="browser",
                payment_timeout=payment_timeout,
            )
        ),
        Billing(billing),
        yes=yes,
    )
    try:
        data = (await cloud_numbers.retrieve(item.id, billing=mode)).data
    except Exception as exc:
        reason = str(exc) if isinstance(exc, MobilerunError) else type(exc).__name__
        click.echo(f"{item.id} created; details unavailable ({reason})", err=True)
        if as_json:
            click.echo(json.dumps({"id": item.id, "status": "unknown"}))
        else:
            render.success(f"Created number {item.id}")
        return
    entry = {"id": item.id, "status": data.state, "phone_number": data.phone_number}
    if as_json:
        click.echo(json.dumps(entry))
    else:
        render.success(
            f"Created number {data.phone_number or item.id} ({item.id}; {data.state})"
        )


@number.command("included", help="Show remaining included numbers.")
@click.option("--country", default=None, help="Country code for the number.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@billing_option
@coro
async def number_included(country: str | None, as_json: bool, billing: str) -> None:
    if country is None:
        countries = (await cloud_numbers.countries(billing=Billing(billing))).data.items
        counts = await _included_counts([item.country for item in countries], billing)
        entries = [
            {"country": item.country, "remaining": count}
            for item, count in zip(countries, counts, strict=True)
        ]
        if as_json:
            click.echo(json.dumps(entries, ensure_ascii=False))
        elif not entries:
            render.empty("countries")
        else:
            render.heading(len(entries), "country")
            render.table(
                [
                    [
                        item.name,
                        item.country,
                        "unavailable" if count is None else f"{count} remaining",
                    ]
                    for item, count in zip(countries, counts, strict=True)
                ]
            )
        return
    remaining = await PhoneNumber.included_available(country, billing=billing)
    if as_json:
        click.echo(json.dumps({"country": country, "remaining": remaining}))
    else:
        render.kv([(f"Included numbers ({country})", f"{remaining} remaining")])


@number.command("release", help="Release phone number IDs.")
@click.option("--json", "as_json", is_flag=True)
@click.option("--yes", "-y", is_flag=True, help="Confirm without prompting.")
@click.argument(
    "number_id",
    cls=PromptingArgument,
    choices=numbers,
    metavar="NUMBER...",
    nargs=-1,
    required=True,
)
@billing_option
@coro
async def number_release(
    number_id: tuple[str, ...], billing: str, yes: bool, as_json: bool
) -> None:
    async def release(item_id: str) -> str:
        result = await PhoneNumber(item_id, billing=Billing(billing)).release()
        return result.data.state

    await cleanup(
        number_id,
        action="Release",
        noun="number",
        billing=billing,
        yes=yes,
        as_json=as_json,
        operation=release,
    )


@number.command("bind", help="Bind NUMBER to cloud phone DEVICE.")
@click.argument("number_id", cls=PromptingArgument, choices=numbers, metavar="NUMBER")
@click.argument(
    "device_id", cls=PromptingArgument, choices=cloud_devices, metavar="DEVICE"
)
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@billing_option
@coro
async def number_bind(
    number_id: str, device_id: str, as_json: bool, billing: str
) -> None:
    mode = Billing(billing)
    async with cloud_client(billing=mode) as client:
        info = await devices.capabilities(device_id, billing=mode, client=client)
        if not info.capabilities.telephony:
            raise click.ClickException("This device does not support cloud telephony")
        current = await devices.number(device_id, billing=mode, client=client)
        if current.binding is not None:
            raise click.ClickException(
                f"Device {device_id} already has a phone number; unbind it first."
            )
        await devices.bind_number(device_id, number_id, billing=mode, client=client)
    if as_json:
        click.echo(
            json.dumps(
                {
                    "id": number_id,
                    "status": "bound",
                    "number_id": number_id,
                    "device_id": device_id,
                }
            )
        )
    else:
        render.success(f"Bound number {number_id} to device {device_id}")


@number.command("unbind", help="Unbind numbers from their cloud phones.")
@click.argument(
    "number_id",
    cls=PromptingArgument,
    choices=numbers,
    metavar="NUMBER...",
    nargs=-1,
    required=True,
)
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@click.option("--yes", "-y", is_flag=True, help="Confirm without prompting.")
@billing_option
@coro
async def number_unbind(
    number_id: tuple[str, ...], as_json: bool, billing: str, yes: bool
) -> None:
    async def unbind(item_id: str) -> str:
        mode = Billing(billing)
        async with cloud_client(billing=mode) as client:
            result = await cloud_numbers.retrieve(item_id, billing=mode, client=client)
            binding = result.data.device_binding
            if binding is None:
                raise click.ClickException(
                    f"Number {item_id} is not bound to a device."
                )
            await devices.unbind_number(binding.device_id, billing=mode, client=client)
        return "unbound"

    await cleanup(
        number_id,
        action="Unbind",
        noun="number",
        billing=billing,
        yes=yes,
        as_json=as_json,
        operation=unbind,
    )


@number.command("code", help="Wait for an SMS code from NUMBER, the phone number ID.")
@click.argument("number_id", cls=PromptingArgument, choices=numbers, metavar="NUMBER")
@click.option("--sender", help="Filter by sender.")
@click.option(
    "--timeout",
    type=click.FloatRange(min=0, min_open=True),
    default=60,
    help="Maximum wait in seconds.",
)
@billing_option
@coro
async def number_sms(
    number_id: str, sender: str | None, timeout: float, billing: str
) -> None:
    click.echo(f"Waiting for SMS on {number_id} ({timeout:g}s)…", err=True)
    try:
        code = await PhoneNumber(number_id, billing=Billing(billing)).wait_for_code(
            sender=sender, timeout=timeout
        )
    except (TimeoutError, ExtensionActionError) as exc:
        if isinstance(exc, ExtensionActionError) and exc.cause_type != "TimeoutError":
            raise
        try:
            data = (
                await cloud_numbers.retrieve(number_id, billing=Billing(billing))
            ).data
            address = data.phone_number or number_id
        except Exception:
            address = number_id
        raise click.ClickException(
            f"Timed out waiting for SMS on {address} after {timeout:g}s (sender: {sender or 'any'}). Retry with a longer --timeout."
        ) from exc
    click.echo(code)
