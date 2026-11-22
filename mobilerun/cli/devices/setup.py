import traceback

import click
from async_adbutils import adb
from mobilerun_core_local.driver.android.portal import PORTAL_PACKAGE_NAME
from rich.console import Console

from mobilerun.cli.base.run import coro
from mobilerun.cli.base.errors import render_error
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.clients.portal.install import install_portal
from mobilerun.errors import DeviceNotFound, PortalApkNotFound, PortalInstallFailed

console = Console()


@click.command()
@click.option(
    "--path",
    help="Path to the Mobilerun Portal APK to install on the device. If not provided, the latest portal apk version will be downloaded and installed.",
    default=None,
)
@click.option(
    "--portal-version",
    "-pv",
    help="Specific Portal version to install (e.g., 0.4.7)",
    default=None,
)
@click.option(
    "--latest",
    is_flag=True,
    help="Install latest Portal instead of compatible version",
    default=False,
)
@click.option(
    "--debug", is_flag=True, help="Enable verbose debug logging", default=False
)
@device_command
@coro
async def setup(
    target: DeviceTarget,
    path: str | None,
    portal_version: str | None,
    latest: bool,
    debug: bool,
) -> None:
    """Install and enable the Mobilerun Portal on a device."""
    if target.kind != "android":
        raise click.ClickException(
            "device setup currently supports Android targets only."
        )
    try:
        device_obj = await adb.device(target.identifier)
        if not device_obj:
            raise DeviceNotFound(f"Could not get device object for {target}")

        if path:
            console.print(f"[bold blue]Using provided APK:[/] {path}")
        elif latest:
            console.print("[bold blue]Downloading latest Portal APK...[/]")

        result = await install_portal(
            device_obj,
            apk_path=path,
            version=portal_version,
            latest=latest,
            debug=debug,
        )
        if result.apk_source:
            console.print(
                f"[bold blue]Step 1/2: Installing APK:[/] {result.apk_source}"
            )
            console.print("[bold green]Installation successful![/]")
            console.print("[bold blue]Step 2/2: Enabling accessibility service[/]")
        if result.accessibility_enabled:
            console.print("[green]Accessibility service enabled successfully![/]")
            console.print(
                "\n[bold green]Setup complete![/] The Mobilerun Portal is now installed and ready to use."
            )
        elif result.needs_manual_accessibility:
            console.print(
                f"[yellow]Could not automatically enable accessibility service: {result.accessibility_error or 'the service did not respond in time'}[/]"
            )
            console.print(
                "[yellow]Opening accessibility settings for manual configuration...[/]"
            )
            console.print(
                "\n[yellow]Please complete the following steps on your device:[/]"
            )
            console.print(
                f"1. Find [bold]{PORTAL_PACKAGE_NAME}[/] in the accessibility services list"
            )
            console.print("2. Tap on the service name")
            console.print("3. Toggle the switch to [bold]ON[/] to enable the service")
            console.print("4. Accept any permission dialogs that appear")
            console.print(
                "\n[bold green]APK installation complete![/] Please manually enable the accessibility service using the steps above."
            )
    except (PortalApkNotFound, PortalInstallFailed, DeviceNotFound) as exc:
        if isinstance(exc, PortalApkNotFound):
            console.print(f"[bold red]Error:[/] {exc}")
        elif isinstance(exc, PortalInstallFailed):
            if not (path or portal_version or latest) and str(exc) == "Setup failed":
                console.print(
                    "[bold red]Setup failed.[/] Run 'mobilerun doctor' for diagnostics."
                )
            else:
                if exc.apk_path:
                    console.print(
                        f"[bold blue]Step 1/2: Installing APK:[/] {exc.apk_path}"
                    )
                console.print(f"[bold red]Installation failed:[/] {exc}")
        else:
            console.print(f"[bold red]Error:[/] {exc}")
        if debug:
            traceback.print_exc()
        raise click.exceptions.Exit(1) from exc
    except Exception as exc:
        render_error(exc, debug=debug)
        raise click.exceptions.Exit(1) from exc
    if not result.installed or not result.accessibility_enabled:
        raise click.exceptions.Exit(1)
