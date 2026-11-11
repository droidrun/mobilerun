"""Device discovery, connection and Portal setup commands."""

import asyncio

import click
from async_adbutils import adb
from mobilerun_core_local.driver.android.portal import (
    ping_portal,
    ping_portal_content,
    ping_portal_tcp,
)

from mobilerun.cli.device_commands import _teardown_android, connect_driver, device_cli
from mobilerun.cli.options import coro, device_options, emit, json_option
from mobilerun.cli.selection import core_client, resolve_cloud_api_key, select_device
from mobilerun.cli.setup import _setup_portal


@device_cli.command("list")
@device_options
@json_option
@click.option("-a", "--all", "all_devices", is_flag=True, help="List local and cloud devices.")
@coro
async def list_devices(
    device, config_path, tcp, platform, target, api_url, verbose, json_output, all_devices
):
    """List devices in the selected target."""
    scope = "all" if all_devices else target
    if scope in {"cloud", "all"} and not resolve_cloud_api_key():
        raise click.ClickException(
            "No cloud API key found. Set MOBILERUN_CLOUD_API_KEY or run `mobilerun auth login`."
        )
    devices = await asyncio.to_thread(core_client(api_url).list_devices, scope=scope)
    rows = [{"device_id": item.device_id, "capabilities": item.capabilities} for item in devices]
    if json_output:
        emit(rows, json_output=True)
    elif rows:
        for row in rows:
            emit(row["device_id"])
    else:
        click.echo("No devices found.", err=True)


@device_cli.command()
@click.argument("serial", required=False)
@device_options
@coro
async def connect(serial, device, config_path, tcp, platform, target, api_url, verbose):
    """Connect to an Android device over TCP/IP."""
    selection = device or serial
    if not selection:
        raise click.UsageError("Supply SERIAL or --device")
    if target == "cloud" or platform == "ios":
        raise click.UsageError("TCP connect requires a local Android device")
    result = await adb.connect(selection)
    if "connected to" not in result:
        raise click.ClickException(f"Failed to connect to {selection}: {result}")
    click.echo(f"Successfully connected to {selection}", err=True)


@device_cli.command()
@click.argument("serial", required=False)
@device_options
@coro
async def disconnect(serial, device, config_path, tcp, platform, target, api_url, verbose):
    """Disconnect an Android TCP/IP device."""
    selection = device or serial
    if not selection:
        raise click.UsageError("Supply SERIAL or --device")
    if target == "cloud" or platform == "ios":
        raise click.UsageError("TCP disconnect requires a local Android device")
    success = await adb.disconnect(selection, raise_error=True)
    click.echo(
        f"Successfully disconnected from {selection}"
        if success
        else f"Device {selection} was not connected",
        err=True,
    )


@device_cli.command()
@device_options
@click.option("--apk", type=click.Path(exists=True, dir_okay=False), help="Portal APK file.")
@click.option("--portal-version", help="Specific Portal version.")
@click.option("--latest", is_flag=True, help="Install the latest Portal.")
@coro
async def setup(
    device, config_path, tcp, platform, target, api_url, verbose, apk, portal_version, latest
):
    """Install and enable the Portal on a local Android device."""
    if sum(bool(x) for x in (apk, portal_version, latest)) > 1:
        raise click.UsageError("Choose one of --apk, --portal-version or --latest")
    selection = await select_device(device, config_path, tcp, platform, target)
    if selection.backend != "local-android-adb":
        raise click.UsageError("Portal setup requires a local Android ADB device")
    if not await _setup_portal(apk, selection.device, bool(verbose), latest, portal_version):
        raise click.ClickException("Portal setup failed; run `mobilerun doctor` for diagnostics")


@device_cli.command()
@device_options
@coro
async def ping(device, config_path, tcp, platform, target, api_url, verbose):
    """Check whether a device and its Portal are reachable."""
    selection = await select_device(device, config_path, tcp, platform, target)
    if selection.backend == "local-android-adb":
        obj = await adb.device(selection.device)
        if not obj:
            raise click.ClickException(f"Could not find device {selection.device}")
        await ping_portal(obj, bool(verbose))
        check = ping_portal_tcp if selection.config.device.use_tcp else ping_portal_content
        await check(obj, bool(verbose))
    else:
        driver, _ = await connect_driver(
            device, config_path, tcp, platform, target, api_url, verbose
        )
        try:
            await driver.get_ui_tree()
        finally:
            await _teardown_android(driver)
    click.echo("Device is reachable and serving the UI tree.", err=True)
