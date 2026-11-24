"""Explicit Connect proxy commands; credentials are never printed."""

import json
import sys
from contextlib import redirect_stdout

import click

from mobilerun.cli.base import menu, render
from mobilerun.cli.base.billing import billing_option
from mobilerun.cli.base.checkout import complete_checkout
from mobilerun.cli.base.choices import cloud_devices, proxies
from mobilerun.cli.base.cleanup import cleanup
from mobilerun.cli.base.menu import MenuGroup, PromptingArgument
from mobilerun.cli.base.prompts import confirm_prompt_async
from mobilerun.cli.base.run import coro
from mobilerun.cli.base.safety import confirm_action_async
from mobilerun.clients.cloud.proxies import (
    buy_proxy,
    cancel_proxy,
    get_proxy,
    list_proxies,
)
from mobilerun.config.settings import get_settings
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.proxy import CloudFixedProxy, UnmanagedProxyConflict
from mobilerun.models.billing import Billing


@click.group(cls=MenuGroup)
def proxy() -> None:
    """Manage Connect proxies."""


@proxy.command("release", help="Release a proxy; phones using this proxy lose it.")
@click.argument(
    "proxy_id", cls=PromptingArgument, choices=proxies, nargs=-1, required=True
)
@click.option("--yes", "-y", is_flag=True, help="Confirm without prompting.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@billing_option
@coro
async def proxy_release(
    proxy_id: tuple[str, ...], yes: bool, as_json: bool, billing: str
) -> None:
    async def release(item_id: str) -> str:
        await cancel_proxy(item_id, billing=Billing(billing))
        return "released"

    await cleanup(
        proxy_id,
        action="Release",
        noun="proxy",
        billing=billing,
        yes=yes,
        as_json=as_json,
        operation=release,
        warning=(
            "; phones using it lose it"
            if len(set(proxy_id)) == 1
            else "; phones using them lose them"
        ),
    )


@proxy.command("bind", help="Bind an existing Connect proxy to a cloud phone.")
@click.option("--json", "as_json", is_flag=True)
@click.argument("proxy_id", cls=PromptingArgument, choices=proxies)
@click.argument(
    "device_id",
    cls=PromptingArgument,
    prompt_label="Cloud device ID",
    choices=cloud_devices,
)
@click.option(
    "--force", is_flag=True, help="Explicitly replace the phone's current proxy."
)
@billing_option
@coro
async def proxy_bind(
    proxy_id: str, device_id: str, force: bool, billing: str, as_json: bool
) -> None:
    phone = CloudDevice(device_id=device_id, billing=Billing(billing))
    extension = CloudFixedProxy(proxy_id)
    try:
        click.echo(f"Binding proxy {proxy_id} to {device_id}…", err=True)
        click.echo("Waiting for connection…", err=True)
        try:
            await extension.bind(phone, force=force)
        except UnmanagedProxyConflict as exc:
            if force or not menu.is_interactive():
                raise click.ClickException(
                    str(exc).replace("force=True", "--force")
                ) from exc
            with redirect_stdout(sys.stderr):
                replace = await confirm_prompt_async(
                    f"{device_id} already has an active proxy. Replace it?",
                    default=True,
                )
            if not replace:
                click.echo("Cancelled.", err=True)
                return
            await extension.bind(phone, force=True)
        status = await extension.status()
        state = "connected" if status.connected else "disconnected"
        if as_json:
            click.echo(
                json.dumps({"id": proxy_id, "status": state, "device_id": device_id})
            )
        elif status.connected:
            render.success(f"Bound proxy {proxy_id} to {device_id}")
        else:
            render.warn(f"Proxy {proxy_id} is disconnected on {device_id}")
    finally:
        # Detach the local relationship while leaving the cloud phone's
        # required proxy configured on the server.
        await phone.close()


@proxy.command("list", help="List Connect proxies.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@billing_option
@coro
async def proxy_list(billing: str, as_json: bool) -> None:
    page = 1
    entries = []
    while True:
        result = await list_proxies(billing=Billing(billing), page=page, page_size=100)
        for item in result.items:
            entries.append(
                {
                    "id": item.id,
                    "country": item.country,
                    "status": item.status,
                    "type": item.type,
                }
            )
        if not result.pagination.has_next:
            break
        page += 1
    if as_json:
        click.echo(json.dumps(entries, ensure_ascii=False))
    elif not entries:
        render.empty("proxies", "Create one: mobilerun proxy buy")
    else:
        render.heading(len(entries), "proxy")
        render.table(
            [
                [
                    render.secondary(entry["id"]),
                    render.secondary(entry["country"]),
                    render.status(entry["status"]),
                    render.secondary(entry["type"]),
                ]
                for entry in entries
            ]
        )


@proxy.command("status", help="Show status for PROXY_ID, the proxy ID.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@click.argument("proxy_id", cls=PromptingArgument, choices=proxies)
@billing_option
@coro
async def proxy_status(proxy_id: str, billing: str, as_json: bool) -> None:
    result = await get_proxy(proxy_id, billing=Billing(billing))
    if as_json:
        click.echo(
            json.dumps(
                {
                    "id": result.id,
                    "country": result.country,
                    "status": result.status,
                    "type": result.type,
                }
            )
        )
        return
    render.table(
        [
            [
                render.secondary(proxy_id),
                render.secondary(result.country),
                render.status(result.status),
                render.secondary(getattr(result, "type", None)),
            ]
        ]
    )


@proxy.command("buy", help="Buy a Connect proxy.")
@click.option("--json", "as_json", is_flag=True)
@click.option("--yes", "-y", is_flag=True, help="Confirm without prompting.")
@click.option(
    "--payment-timeout",
    type=click.FloatRange(min=0, min_open=True),
    default=600,
    help="Payment wait timeout in seconds.",
)
@click.option("--country", required=True, help="Country code for the proxy.")
@click.option(
    "--type",
    "proxy_type",
    type=click.Choice(["residential", "dedicated_residential", "mobile"]),
    default="residential",
    help="Proxy network type. Use dedicated_residential for a fixed IP; pass the printed ID to CloudFixedProxy.",
)
@click.option(
    "--billing",
    type=click.Choice(["included", "paid"]),
    default=lambda: get_settings().MOBILERUN_BILLING.value,
    show_default="configured billing",
    help="Proxy purchases require paid; included is not supported.",
)
@coro
async def proxy_buy(
    country: str,
    proxy_type: str,
    billing: str,
    yes: bool,
    payment_timeout: float,
    as_json: bool,
) -> None:
    if billing != "paid":
        raise click.BadParameter(
            "Proxy purchases require --billing paid", param_hint="--billing"
        )
    await confirm_action_async(
        "buy proxy", billing, "This may incur charges.", yes=yes, err=True
    )
    if proxy_type not in ("residential", "dedicated_residential", "mobile"):
        raise click.BadParameter("Invalid proxy type", param_hint="--type")
    with redirect_stdout(sys.stderr):
        result = await complete_checkout(
            buy_proxy(
                country,
                type=proxy_type,
                billing=Billing(billing),
                checkout="browser",
                payment_timeout=payment_timeout,
            )
        )
    if as_json:
        click.echo(
            json.dumps({"id": result.id, "status": result.status, "country": country})
        )
    else:
        render.success(f"Bought proxy {result.id} ({country})")
