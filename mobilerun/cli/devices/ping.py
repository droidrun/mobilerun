
import click
from async_adbutils import adb
from mobilerun_core_local.driver.android.portal import (
    ping_portal,
    ping_portal_content,
    ping_portal_tcp,
)

from mobilerun.cli.base import render
from mobilerun.cli.base.errors import render_error
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.clients.portal.http import (
    create_android_portal_http_driver,
    is_portal_http_url,
)


@click.command()
@click.option("--debug/--no-debug", default=None, help="Enable verbose debug logging")
@device_command
@coro
async def ping(
    target: DeviceTarget,
    debug: bool | None,
) -> None:
    """Ping a device to check if it is ready and accessible."""
    # Handle None defaults
    debug_mode = debug if debug is not None else False
    config = click.get_current_context().meta["device_config"]
    use_tcp_mode = target.tcp
    if target.kind != "android":
        raise click.ClickException(
            "device ping currently supports Android targets only."
        )

    try:
        if is_portal_http_url(target.identifier):
            token = config.device.resolve_auth_token()
            driver = create_android_portal_http_driver(target.identifier, token)
            try:
                await driver.connect()
                await driver.get_ui_tree()
            finally:
                await driver.close()
            render.success(f"{target}: Portal HTTP is reachable.")
            return

        device_obj = await adb.device(target.identifier)
        if not device_obj:
            raise click.ClickException(f"{target}: device not found")

        await ping_portal(device_obj, debug_mode)

        if use_tcp_mode:
            await ping_portal_tcp(device_obj, debug_mode)
        else:
            await ping_portal_content(device_obj, debug_mode)

        render.success(f"{target}: Portal is accessible.")
    except Exception as e:
        render_error(e, debug=debug_mode)
        raise click.exceptions.Exit(1) from e
