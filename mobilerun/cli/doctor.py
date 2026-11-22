import asyncio

import click
from async_adbutils import AdbDevice
from mobilerun_core_local.driver.android.portal import (
    PORTAL_PACKAGE_NAME,
    check_portal_accessibility,
    enable_portal_accessibility,
)

from mobilerun.cli.base import render
from mobilerun.cli.base.errors import render_error
from mobilerun.cli.base.menu import is_interactive
from mobilerun.cli.base.prompts import SelectChoice, select_prompt
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.clients.portal.http import is_portal_http_url
from mobilerun.clients.portal.install import install_portal
from mobilerun.diagnostics.checks import (
    check_accessibility,
    check_adb,
    check_config,
    check_content_provider,
    check_device,
    check_keyboard,
    check_portal_installed,
    check_portal_state,
    check_portal_version,
    check_screenshot,
    check_sdk_version,
    check_tcp,
)
from mobilerun.diagnostics.results import CheckResult, Status


class DoctorCommand(click.Command):
    def parse_args(self, ctx, args):
        if not args and not ctx.resilient_parsing and is_interactive():
            selected = select_prompt(
                "Doctor",
                [
                    SelectChoice("diagnose", "Diagnose"),
                    SelectChoice("fix", "Diagnose and repair"),
                ],
            )
            if selected is None:
                ctx.exit(0)
            if selected == "fix":
                args = ["--fix"]
        return super().parse_args(ctx, args)


@click.command(cls=DoctorCommand)
@click.option("--debug/--no-debug", default=None, help="Enable verbose debug output")
@click.option(
    "--fix", is_flag=True, help="Repair Portal installation and accessibility."
)
@device_command(optional=True)
@coro
async def doctor(
    target: DeviceTarget,
    debug: bool | None,
    fix: bool,
) -> None:
    """Check system health and diagnose issues."""
    serial = target.identifier
    if serial is None:
        render.warn(
            "No device given: skipping device checks (use --android SERIAL ...)"
        )
    else:
        if target.kind == "android" and is_portal_http_url(serial):
            render.warn(
                "doctor: device checks for Portal HTTP targets aren't supported yet; ran local checks"
            )
            serial = None
        elif target.kind != "android":
            render.warn("doctor: device checks support Android only; ran local checks")
            serial = None
    await run_doctor(serial, debug if debug is not None else False, fix=fix)


def _print_result(result: CheckResult, debug: bool = False) -> None:
    output = {
        Status.PASS: render.success,
        Status.WARN: render.warn,
        Status.FAIL: render.error,
    }
    output[result.status](f"{result.name}: {result.message}")
    if debug and result.detail:
        render.kv([("  Detail", result.detail)])


# ── Auto-fix Functions ───────────────────────────────────────────


async def fix_enable_accessibility(device: AdbDevice, debug: bool) -> bool:
    """Enable accessibility service. Returns True on success."""
    render.kv([("Action", "Enabling accessibility service")])
    try:
        await enable_portal_accessibility(device)
        # Verify
        enabled = await check_portal_accessibility(device, debug=debug)
        if enabled:
            render.success("Accessibility enabled")
            return True
        else:
            render.warn("Auto-enable may have failed; opening settings")
            await device.shell("am start -a android.settings.ACCESSIBILITY_SETTINGS")
            render.warn(f"Please enable {PORTAL_PACKAGE_NAME} manually")
            return False
    except Exception as e:
        render_error(e, debug=debug)
        try:
            await device.shell("am start -a android.settings.ACCESSIBILITY_SETTINGS")
            render.warn(f"Please enable {PORTAL_PACKAGE_NAME} manually")
        except Exception:
            pass
        return False


# ── Orchestrator ─────────────────────────────────────────────────


async def run_doctor(
    device_serial: str | None = None,
    debug: bool = False,
    *,
    fix: bool = False,
) -> None:
    """Run all doctor checks and auto-fix where possible."""
    debug = debug or False

    render.kv([("Mobilerun Doctor", "system health")])

    results: list[CheckResult] = []

    # 1. SDK Version
    r = await check_sdk_version(debug)
    _print_result(r, debug)
    results.append(r)

    # 2. Config
    r_config, config = await check_config(debug)
    _print_result(r_config, debug)
    results.append(r_config)

    # 3. ADB
    r = await check_adb(debug)
    _print_result(r, debug)
    results.append(r)
    if r.status == Status.FAIL:
        _print_summary(results)
        return

    if device_serial is None:
        _print_summary(results)
        return

    # 5. Device
    r_device, device = await check_device(device_serial, debug)
    _print_result(r_device, debug)
    results.append(r_device)
    if device is None:
        render.warn("Skipping device-dependent checks (no device)")
        _print_summary(results)
        return

    # 6. Portal Installed
    r_portal, is_installed = await check_portal_installed(device, debug)
    _print_result(r_portal, debug)
    if not is_installed and fix:
        # Auto-fix: run setup
        await install_portal(device, debug=debug)
        render.success(
            "Setup complete! The Mobilerun Portal is now installed and ready to use."
        )
        # Re-check
        r_portal, is_installed = await check_portal_installed(device, debug)
        if not is_installed:
            results.append(r_portal)
            _print_summary(results)
            return
    results.append(r_portal)
    if not is_installed:
        _print_summary(results)
        return

    # 7. Portal Version
    r_version, installed_ver, _, _ = await check_portal_version(device, debug)
    _print_result(r_version, debug)
    if fix and r_version.status == Status.WARN and installed_ver:
        # Auto-fix: run setup to install compatible version
        render.kv([("Action", "Updating portal")])
        await install_portal(device, debug=debug, in_place=True)
        render.success(
            "Setup complete! The Mobilerun Portal is now installed and ready to use."
        )
        # Wait for portal service to initialize after install
        render.kv([("Action", "Waiting for portal to start")])
        await asyncio.sleep(3)
        # Re-check
        r_version, _, _, _ = await check_portal_version(device, debug)
        if r_version.status == Status.PASS:
            render.success("Portal updated")
        else:
            render.warn("Update may not have resolved the version mismatch")
    results.append(r_version)

    # 8. Accessibility
    r_a11y, a11y_enabled = await check_accessibility(device, debug)
    if not a11y_enabled:
        _print_result(r_a11y, debug)
        # Auto-fix: enable accessibility
        success = await fix_enable_accessibility(device, debug) if fix else False
        if success:
            r_a11y = CheckResult("Accessibility", Status.PASS, "enabled (auto-fixed)")
            _print_result(r_a11y, debug)
    else:
        _print_result(r_a11y, debug)
    results.append(r_a11y)

    # 9. Content Provider — connectivity, state, screenshot
    r = await check_content_provider(device, debug)
    _print_result(r, debug)
    results.append(r)

    r_cp_state, cp_portal = await check_portal_state(device, False, debug)
    r_cp_state.name = "State (content)"
    _print_result(r_cp_state, debug)
    results.append(r_cp_state)

    if cp_portal and r_cp_state.status == Status.PASS:
        r = await check_screenshot(cp_portal, debug)
        r.name = "Screenshot (content)"
        _print_result(r, debug)
        results.append(r)

    # 10. TCP — connectivity, state, screenshot
    r = await check_tcp(device, debug, fix=fix)
    _print_result(r, debug)
    results.append(r)

    if r.status == Status.PASS and fix:
        r_tcp_state, tcp_portal = await check_portal_state(device, True, debug)
        r_tcp_state.name = "State (tcp)"
        _print_result(r_tcp_state, debug)
        results.append(r_tcp_state)

        if tcp_portal and r_tcp_state.status == Status.PASS:
            r = await check_screenshot(tcp_portal, debug)
            r.name = "Screenshot (tcp)"
            _print_result(r, debug)
            results.append(r)

    # 12. Keyboard
    r = await check_keyboard(device, debug)
    _print_result(r, debug)
    results.append(r)

    _print_summary(results)


def _print_summary(results: list[CheckResult]) -> None:
    """Print final summary."""
    fails = [r for r in results if r.status == Status.FAIL]
    warns = [r for r in results if r.status == Status.WARN]

    if not fails and not warns:
        render.success("All checks passed.")
    else:
        if fails:
            render.error(f"{len(fails)} {'issue' if len(fails) == 1 else 'issues'}")
            for r in fails:
                render.error(f"{r.name}: {r.message}")
                if r.detail:
                    render.kv([("  Detail", r.detail)])
        if warns:
            render.warn(f"{len(warns)} {'warning' if len(warns) == 1 else 'warnings'}")
            for r in warns:
                render.warn(f"{r.name}: {r.message}")
                if r.detail:
                    render.kv([("  Detail", r.detail)])
    if fails:
        raise click.exceptions.Exit(1)
