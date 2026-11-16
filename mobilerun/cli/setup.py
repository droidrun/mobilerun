"""
Mobilerun CLI - Command line interface for controlling Android devices through LLM agents.
"""

import asyncio
import os
from contextlib import nullcontext

from async_adbutils import adb
from mobilerun_core_local.driver.android.portal import (
    DOWNLOAD_BASE,
    PORTAL_PACKAGE_NAME,
    check_portal_accessibility,
    download_portal_apk,
    download_versioned_portal_apk,
    enable_portal_accessibility,
    portal_content_uri,
    setup_portal,
)
from rich.console import Console

from mobilerun.agent.utils.android_device import (
    AndroidDeviceSelectionError,
    resolve_android_serial,
)
from mobilerun.agent.utils.portal_setup import portal_version_kwargs

console = Console(stderr=True)


async def _wait_for_portal_service(
    device_obj, timeout: float = 30.0, interval: float = 1.0
) -> bool:
    """Wait until Portal's accessibility service responds.

    Android drops the service when it is enabled before a fresh install is
    registered, so it is enabled again while waiting. ``/ping`` answers once
    the service is bound, even without an active window. A busy device can
    take well over 10s to bind it.
    """
    uri = portal_content_uri(PORTAL_PACKAGE_NAME, "ping")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            result = await device_obj.shell(f"content query --uri {uri}")
            if '"status":"success"' in result:
                return True
            if not await check_portal_accessibility(device_obj):
                await enable_portal_accessibility(device_obj)
        except Exception:
            pass
        await asyncio.sleep(interval)
    return False


async def _setup_portal(
    path: str | None,
    device: str | None,
    debug: bool,
    latest: bool = False,
    specific_version: str | None = None,
    in_place: bool = False,
):
    """Internal async function to install and enable the Mobilerun Portal on a device."""
    try:
        if not device:
            try:
                device = await resolve_android_serial()
            except AndroidDeviceSelectionError as e:
                console.print(f"[bold red]Error:[/] {e}")
                return False
            console.print(f"[blue]Using device:[/] {device}")

        device_obj = await adb.device(device)
        if not device_obj:
            console.print(
                f"[bold red]Error:[/] Could not get device object for {device}"
            )
            return False

        # CLI-specific options: path, specific_version, latest
        if path:
            console.print(f"[bold blue]Using provided APK:[/] {path}")
            apk_context = nullcontext(path)
        elif specific_version:
            version = specific_version.lstrip("v")
            version = f"v{version}"
            download_base = DOWNLOAD_BASE
            apk_context = download_versioned_portal_apk(version, download_base, debug)
        elif latest:
            console.print("[bold blue]Downloading latest Portal APK...[/]")
            apk_context = download_portal_apk(debug)
        else:
            # Default: delegate to shared setup_portal()
            success = await setup_portal(
                device_obj,
                debug,
                **portal_version_kwargs(setup_portal, in_place=in_place),
            )
            if success:
                console.print(
                    "\n[bold green]Setup complete![/] The Mobilerun Portal is now installed and ready to use."
                )
            else:
                console.print(
                    "[bold red]Setup failed.[/] Run 'mobilerun doctor' for diagnostics."
                )
            return bool(success)

        # Install from explicit path/version/latest
        with apk_context as apk_path:
            if not os.path.exists(apk_path):
                console.print(f"[bold red]Error:[/] APK file not found at {apk_path}")
                return False

            console.print(f"[bold blue]Step 1/2: Installing APK:[/] {apk_path}")
            try:
                await device_obj.install(
                    apk_path,
                    nolaunch=True,
                    uninstall=True,
                    flags=["-g"],
                    silent=not debug,
                )
            except Exception as e:
                console.print(f"[bold red]Installation failed:[/] {e}")
                return False

            console.print("[bold green]Installation successful![/]")

            console.print("[bold blue]Step 2/2: Enabling accessibility service[/]")

            try:
                await enable_portal_accessibility(device_obj)
                if not await _wait_for_portal_service(device_obj):
                    raise RuntimeError("the service did not respond in time")

                console.print("[green]Accessibility service enabled successfully![/]")
                console.print(
                    "\n[bold green]Setup complete![/] The Mobilerun Portal is now installed and ready to use."
                )

            except Exception as e:
                console.print(
                    f"[yellow]Could not automatically enable accessibility service: {e}[/]"
                )
                console.print(
                    "[yellow]Opening accessibility settings for manual configuration...[/]"
                )

                await device_obj.shell(
                    "am start -a android.settings.ACCESSIBILITY_SETTINGS"
                )

                console.print(
                    "\n[yellow]Please complete the following steps on your device:[/]"
                )
                console.print(
                    f"1. Find [bold]{PORTAL_PACKAGE_NAME}[/] in the accessibility services list"
                )
                console.print("2. Tap on the service name")
                console.print(
                    "3. Toggle the switch to [bold]ON[/] to enable the service"
                )
                console.print("4. Accept any permission dialogs that appear")

                console.print(
                    "\n[bold green]APK installation complete![/] Please manually enable the accessibility service using the steps above."
                )
                return False
        return True

    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")

        if debug:
            import traceback

            traceback.print_exc()

        return False
