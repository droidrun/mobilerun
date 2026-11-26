"""Cloud mailbox commands."""

import json

import click

from mobilerun.cli.base import render
from mobilerun.cli.base.billing import (
    billing_option,
    invoke_with_billing,
)
from mobilerun.cli.base.checkout import complete_checkout
from mobilerun.cli.base.choices import mailboxes
from mobilerun.cli.base.cleanup import cleanup
from mobilerun.cli.base.menu import MenuGroup, PromptingArgument
from mobilerun.cli.base.run import coro
from mobilerun.clients.cloud import mailboxes as cloud_mailboxes
from mobilerun.clients.cloud.mailboxes import list_mailboxes
from mobilerun.core.mailbox import CloudMailbox as Mailbox
from mobilerun.errors import ExtensionActionError
from mobilerun.errors.base import MobilerunError
from mobilerun.models.billing import Billing


@click.group(cls=MenuGroup)
def mailbox() -> None:
    """Manage cloud email mailboxes."""


@mailbox.command("messages", help="List email message history.")
@click.argument(
    "mailbox_id", cls=PromptingArgument, choices=mailboxes, metavar="MAILBOX"
)
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@billing_option
@coro
async def mailbox_messages(mailbox_id: str, as_json: bool, billing: str) -> None:
    items = await cloud_mailboxes.messages(mailbox_id, billing=Billing(billing))
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
                    entry["received_at"],
                    entry.get("from_address") or "",
                    entry.get("subject") or "",
                    "OTP available" if entry.get("has_otp") else "",
                    (entry.get("preview") or "")[:100],
                ]
                for entry in entries
            ]
        )


@mailbox.command("status", help="Show a mailbox's lifecycle state.")
@click.argument(
    "mailbox_id", cls=PromptingArgument, choices=mailboxes, metavar="MAILBOX"
)
@click.option("--json", "as_json", is_flag=True)
@billing_option
@coro
async def mailbox_status(mailbox_id: str, as_json: bool, billing: str) -> None:
    data = (await cloud_mailboxes.retrieve(mailbox_id, billing=Billing(billing))).data
    entry = {"id": data.id, "status": data.status, "address": data.address}
    if as_json:
        click.echo(json.dumps(entry))
    else:
        render.table(
            [
                [
                    entry["address"] or entry["id"],
                    render.status(entry["status"]),
                    render.secondary(entry["id"]) if entry["address"] else "",
                ]
            ]
        )


@mailbox.command("list", help="List cloud mailboxes.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@billing_option
@coro
async def mailbox_list(billing: str, as_json: bool) -> None:
    items = await list_mailboxes(billing=Billing(billing))
    entries = [
        {"id": item.id, "address": item.address, "status": item.status}
        for item in items
    ]
    if as_json:
        click.echo(json.dumps(entries, ensure_ascii=False))
    elif not entries:
        render.empty("mailboxes", "Create one: mobilerun mailbox create")
    else:
        render.heading(len(entries), "mailbox")
        render.table(
            [
                [
                    str(entry["address"] or entry["id"]),
                    render.status(entry["status"]),
                    render.secondary(entry["id"]) if entry["address"] else "",
                ]
                for entry in entries
            ]
        )


@mailbox.command("create", help="Create a cloud mailbox.")
@click.option("--json", "as_json", is_flag=True)
@click.option(
    "--plan",
    type=click.Choice(["auto", "included", "rent"]),
    default="auto",
    show_default=True,
    help="auto = included first, then rent if billing is paid; included = never rent; rent = always rent (requires paid billing).",
)
@click.option("--yes", "-y", is_flag=True, help="Confirm billing without prompting.")
@click.option(
    "--payment-timeout",
    type=click.FloatRange(min=0, min_open=True),
    default=600,
    help="Payment wait timeout in seconds.",
)
@billing_option
@coro
async def mailbox_create(
    billing: str, yes: bool, payment_timeout: float, plan: str, as_json: bool
) -> None:
    item, mode = await invoke_with_billing(
        lambda attempt: complete_checkout(
            Mailbox(
                billing=attempt,
                plan=plan,
                keep=True,
                checkout="browser",
                payment_timeout=payment_timeout,
            )
        ),
        Billing(billing),
        yes=yes,
        confirm_paid=None if plan == "included" else "Create mailbox",
    )
    try:
        data = (await cloud_mailboxes.retrieve(item.id, billing=mode)).data
    except Exception as exc:
        reason = str(exc) if isinstance(exc, MobilerunError) else type(exc).__name__
        click.echo(f"{item.id} created; details unavailable ({reason})", err=True)
        if as_json:
            click.echo(json.dumps({"id": item.id, "status": "unknown"}))
        else:
            render.success(f"Created mailbox {item.id}")
        return
    entry = {"id": item.id, "status": data.status, "address": data.address}
    if as_json:
        click.echo(json.dumps(entry))
    else:
        render.success(
            f"Created mailbox {data.address or item.id} ({item.id}; {data.status})"
        )


@mailbox.command("included", help="Show remaining included mailboxes.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@billing_option
@coro
async def mailbox_included(as_json: bool, billing: str) -> None:
    remaining = await Mailbox.included_available(billing=billing)
    if as_json:
        click.echo(json.dumps({"remaining": remaining}))
    else:
        render.kv([("Included mailboxes", f"{remaining} remaining")])


@mailbox.command("release", help="Release mailbox IDs.")
@click.option("--json", "as_json", is_flag=True)
@click.option("--yes", "-y", is_flag=True, help="Confirm without prompting.")
@click.argument(
    "mailbox_id",
    cls=PromptingArgument,
    choices=mailboxes,
    metavar="MAILBOX...",
    nargs=-1,
    required=True,
)
@billing_option
@coro
async def mailbox_release(
    mailbox_id: tuple[str, ...], billing: str, yes: bool, as_json: bool
) -> None:
    async def release(item_id: str) -> str:
        result = await Mailbox(item_id, billing=Billing(billing)).release()
        return result.data.status

    await cleanup(
        mailbox_id,
        action="Release",
        noun="mailbox",
        billing=billing,
        yes=yes,
        as_json=as_json,
        operation=release,
    )


@mailbox.command(
    "code", help="Wait for a verification code from MAILBOX, the mailbox ID."
)
@click.argument(
    "mailbox_id", cls=PromptingArgument, choices=mailboxes, metavar="MAILBOX"
)
@click.option("--sender", help="Filter by sender.")
@click.option(
    "--timeout",
    type=click.FloatRange(min=0, min_open=True),
    default=60,
    help="Maximum wait in seconds.",
)
@billing_option
@coro
async def mailbox_code(
    mailbox_id: str, sender: str | None, timeout: float, billing: str
) -> None:
    click.echo(f"Waiting for email on {mailbox_id} ({timeout:g}s)…", err=True)
    try:
        code = await Mailbox(mailbox_id, billing=Billing(billing)).wait_for_code(
            sender=sender, timeout=timeout
        )
    except (TimeoutError, ExtensionActionError) as exc:
        if isinstance(exc, ExtensionActionError) and exc.cause_type != "TimeoutError":
            raise
        try:
            data = (
                await cloud_mailboxes.retrieve(mailbox_id, billing=Billing(billing))
            ).data
            address = data.address or mailbox_id
        except Exception:
            address = mailbox_id
        raise click.ClickException(
            f"Timed out waiting for email on {address} after {timeout:g}s (sender: {sender or 'any'}). Retry with a longer --timeout."
        ) from exc
    click.echo(code)
