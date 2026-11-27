"""Discover existing local and cloud devices without creating resources."""

import logging
from dataclasses import dataclass
from typing import Literal

import click
from async_adbutils import adb

from mobilerun.clients.cloud.devices import list_devices
from mobilerun.clients.local.login_store import resolve_cloud_api_key
from mobilerun.errors.base import CredentialError, MobilerunError
from mobilerun.models.billing import Billing

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveredDevice:
    id: str
    source: Literal["local", "cloud"]
    name: str | None
    state: str | None
    type: str | None

    @property
    def ready(self) -> bool:
        return self.state == ("device" if self.source == "local" else "ready")


async def discover_devices(
    *,
    cloud: bool = True,
    local: bool = True,
    base_url: str | None = None,
    billing: Billing | None = None,
    strict: bool = False,
) -> list[DiscoveredDevice]:
    """List local and cloud devices; strict mode raises cloud errors."""
    devices: list[DiscoveredDevice] = []
    if local:
        try:
            local_devices = await adb.list()
        except Exception:
            logger.warning("Local device discovery failed; use --debug for details.")
            logger.debug("Local discovery failure", exc_info=True)
        else:
            devices.extend(
                DiscoveredDevice(device.serial, "local", None, device.state, None)
                for device in local_devices
            )
    if cloud:
        try:
            api_key = resolve_cloud_api_key()
        except CredentialError as exc:
            if strict:
                raise click.ClickException(str(exc)) from exc
            click.echo(f"Warning: {exc}", err=True)
            return devices
        if not api_key and strict:
            raise click.ClickException(
                "No cloud API key found. Set MOBILERUN_API_KEY or run `mobilerun account login`."
            )
        if api_key:
            try:
                items = await list_devices(
                    api_key=api_key,
                    base_url=base_url,
                    billing=billing or Billing.NO_COMMERCE,
                )
            except Exception as exc:
                if strict:
                    if not isinstance(exc, MobilerunError):
                        raise click.ClickException(
                            "Error listing cloud devices: offline or unavailable (use --debug for details)."
                        ) from exc
                    raise click.ClickException(
                        f"Error listing cloud devices: {exc}"
                    ) from exc
                reason = (
                    str(exc)
                    if isinstance(exc, MobilerunError)
                    else "use --debug for details"
                )
                click.echo(f"Warning: Error listing cloud devices: {reason}", err=True)
                logger.debug("Cloud discovery failure", exc_info=True)
            else:
                devices.extend(
                    DiscoveredDevice(item.id, "cloud", item.name, item.state, item.type)
                    for item in items
                )
    return devices
