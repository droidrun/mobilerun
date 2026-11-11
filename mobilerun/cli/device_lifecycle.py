"""Device discovery and Android lifecycle commands."""

from __future__ import annotations

import click
from async_adbutils import adb

from mobilerun.cli.device_commands import device_cli, probe_device
from mobilerun.cli.device_pair import adb_connect_succeeded
from mobilerun.cli.options import (
    NO_CLOUD_DEVICES_HINT,
    NO_LOCAL_DEVICES_HINT,
    api_url_option,
    config_option,
    convert_api_error,
    coro,
    device_option,
    emit,
    emit_rows,
    examples,
    json_option,
    platform_option,
    redact_secrets,
    target_option,
    verbose_option,
)
from mobilerun.cli.selection import core_client, select_device


@device_cli.command(
    "list", epilog=examples(
        'mobilerun device list',
        'mobilerun device list --all -j',
    ),
)
@target_option
@api_url_option
@verbose_option
@json_option
@click.option("--all", "all_devices", is_flag=True, help="List local and cloud devices.")
def list_devices(target, api_url, verbose, json_output, all_devices):
    """List devices in the selected target."""
    rows, errors = [], []
    scopes = ["local", "cloud"] if all_devices else [target]
    for scope in scopes:
        try:
            rows.extend(_list_target(api_url, scope))
        except Exception as exc:
            if not all_devices:
                raise
            converted = convert_api_error(exc)
            message = converted[1]["error"] if converted else str(exc)
            message = " ".join(redact_secrets(message, click.get_current_context()).split())
            errors.append({"target": scope, "error": message})
            click.echo(f"Warning: {scope}: {' '.join(message.split())}", err=True)
    if json_output:
        emit({"devices": rows, "errors": errors} if all_devices else rows, json_output=True)
    elif rows:
        emit_rows(("DEVICE_ID", "NAME", "STATE", "TYPE"), (
            (row["device_id"], row["name"] or "-", row["state"], row["type"]) for row in rows
        ))
    elif len(errors) < len(scopes):
        hint = NO_CLOUD_DEVICES_HINT if target == "cloud" and not all_devices else NO_LOCAL_DEVICES_HINT
        click.echo("No devices found. " + hint, err=True)
    if len(errors) == len(scopes):
        click.get_current_context().exit(1)


def _list_target(api_url, scope):
    client = core_client(api_url, cloud=scope == "cloud")
    devices = []
    try:
        devices = client.list_devices(filters={"state": None}, scope=scope)
        rows = []
        for device in devices:
            # Core seeds this model from the list response, but has no public
            # metadata accessor for name/state/type on Device yet.
            metadata = getattr(device._conn, "_device_cache", None)
            rows.append({
                "device_id": device.device_id,
                "name": metadata.name if metadata else None,
                "state": metadata.state if metadata else "connected",
                "type": metadata.type if metadata else device.capabilities["backend"],
            })
        return rows
    finally:
        for resource in [*devices, client]:
            close = getattr(resource, "close", None)
            if close:
                close()


@device_cli.command()
@click.argument("serial", metavar="HOST:PORT")
@verbose_option
@coro
async def connect(serial, verbose):
    """Connect to an Android device over TCP/IP."""
    result = await adb.connect(serial)
    if not adb_connect_succeeded(result):
        raise click.ClickException(f"Failed to connect to {serial}: {result}")
    click.echo(f"Successfully connected to {serial}", err=True)


@device_cli.command()
@click.argument("serial", metavar="HOST:PORT")
@verbose_option
@coro
async def disconnect(serial, verbose):
    """Disconnect an Android TCP/IP device."""
    success = await adb.disconnect(serial, raise_error=True)
    click.echo(
        f"Successfully disconnected from {serial}"
        if success else f"Device {serial} was not connected", err=True,
    )


@device_cli.command()
@device_option
@config_option
@verbose_option
@click.option("--apk", type=click.Path(exists=True, dir_okay=False), help="Portal APK file.")
@click.option("--portal-version", help="Specific Portal version.")
@click.option("--latest", is_flag=True, help="Install the latest Portal.")
@coro
async def setup(device, config_path, verbose, apk, portal_version, latest):
    """Install and enable the Portal on a local Android device."""
    from mobilerun.cli.setup import _setup_portal

    if sum(bool(x) for x in (apk, portal_version, latest)) > 1:
        raise click.UsageError("Choose one of --apk, --portal-version or --latest")
    selection = await select_device(device, config_path)
    if selection.backend != "local-android-adb":
        raise click.ClickException("Portal setup requires a local Android ADB device")
    if not await _setup_portal(apk, selection.device, bool(verbose), latest, portal_version):
        raise click.ClickException("Portal setup failed; run `mobilerun doctor` for diagnostics")


@device_cli.command()
@device_option
@target_option
@config_option
@api_url_option
@verbose_option
@platform_option
def ping(**params):
    """Connect to a device and read its UI state once."""
    probe_device(params)
    click.echo("Device is reachable and serving the UI tree.", err=True)


from mobilerun.cli.options import _lazy_imports  # noqa: E402

__getattr__ = _lazy_imports(globals(), {
    '_setup_portal': ('mobilerun.cli.setup', '_setup_portal'),
})

from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from mobilerun.cli.setup import _setup_portal as _setup_portal
