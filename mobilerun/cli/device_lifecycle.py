"""Device discovery and Android lifecycle commands."""

import click
from async_adbutils import adb

from mobilerun.cli.device_commands import device_cli, probe_device
from mobilerun.cli.device_pair import adb_connect_succeeded
from mobilerun.cli.options import (
    NO_LOCAL_DEVICES_HINT,
    api_url_option,
    config_option,
    convert_api_error,
    coro,
    device_option,
    emit,
    emit_rows,
    json_option,
    platform_option,
    target_option,
    verbose_option,
)
from mobilerun.cli.selection import core_client, select_device
from mobilerun.cli.setup import _setup_portal


@device_cli.command(
    "list", epilog='\b\nExamples:\n  mobilerun device list\n  mobilerun device list --all -j',
)
@target_option
@api_url_option
@verbose_option
@json_option
@click.option("--all", "all_devices", is_flag=True, help="List local and cloud devices.")
def list_devices(target, api_url, verbose, json_output, all_devices):
    """List devices in the selected target."""
    rows, errors = [], []
    for scope in (["local", "cloud"] if all_devices else [target]):
        try:
            rows.extend(_list_target(api_url, scope))
        except Exception as exc:
            if not all_devices:
                raise
            converted = convert_api_error(exc)
            message = converted[0] if converted else str(exc)
            errors.append({"target": scope, "error": message})
            click.echo(f"Warning: {scope}: {' '.join(message.split())}", err=True)
    if json_output:
        emit({"devices": rows, "errors": errors} if all_devices else rows, json_output=True)
    elif rows:
        emit_rows(("DEVICE_ID", "NAME", "STATE", "TYPE"), (
            (row["device_id"], row["name"] or "-", row["state"], row["type"]) for row in rows
        ))
    else:
        click.echo("No devices found. " + NO_LOCAL_DEVICES_HINT, err=True)
    if len(errors) == 2:
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
