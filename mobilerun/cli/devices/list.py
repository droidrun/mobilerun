import json

import click

from mobilerun.cli.base import render
from mobilerun.cli.base.billing import billing_option
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.discovery import DiscoveredDevice, discover_devices
from mobilerun.config.loader import ConfigLoader
from mobilerun.config.settings import get_settings
from mobilerun.errors.base import MobilerunError
from mobilerun.models.billing import Billing


def _render_cloud(items: list[DiscoveredDevice], default: str | None = None) -> None:
    if not items:
        render.empty("cloud devices")
        return
    render.heading(len(items), "cloud device")
    rows = []
    for item in items:
        rows.append(
            [
                ("★ " if item.id == default else "") + (item.name or item.id),
                render.secondary(item.id) if item.name else "",
                render.status(item.state),
                render.secondary(item.type),
            ]
        )
    render.table(rows)


@click.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@click.option(
    "--cloud", is_flag=True, default=False, help="Only list Mobilerun cloud devices"
)
@click.option(
    "--base-url",
    default=None,
    help=f"Cloud API base URL (default {get_settings().MOBILERUN_CLOUD_BASE_URL})",
)
@billing_option
@coro
async def devices(
    cloud: bool, base_url: str | None, billing: str, as_json: bool
) -> None:
    """List connected local devices and authenticated cloud devices."""
    if cloud:
        items = await discover_devices(
            local=False,
            cloud=True,
            base_url=base_url,
            billing=Billing(billing),
            strict=True,
        )
    else:
        try:
            items = await discover_devices(base_url=base_url, billing=Billing(billing))
        except MobilerunError as exc:
            raise click.ClickException(f"Error listing local devices: {exc}") from exc
    default = ConfigLoader.load(None).device.serial
    if as_json:
        payload = [
            {
                "id": item.id,
                "source": item.source,
                "default": item.id == default,
                **(
                    {"name": item.name, "state": item.state, "type": item.type}
                    if item.source == "cloud"
                    else {"state": item.state}
                ),
            }
            for item in items
        ]
        click.echo(json.dumps(payload, ensure_ascii=False))
        return
    local_items = [item for item in items if item.source == "local"]
    cloud_items = [item for item in items if item.source == "cloud"]
    if not cloud:
        if local_items:
            render.heading(len(local_items), "local device")
            render.table(
                [
                    [
                        render.secondary(
                            ("★ " if item.id == default else "") + item.id
                        ),
                        render.secondary(item.state),
                    ]
                    for item in local_items
                ]
            )
        else:
            render.empty("local devices")
    if cloud or cloud_items:
        if not cloud:
            click.echo()
        _render_cloud(cloud_items, default)
