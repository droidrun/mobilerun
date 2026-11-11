"""Click wiring for health checks."""

import click

from mobilerun.cli.doctor import Status, console, run_doctor
from mobilerun.cli.options import (
    CONTEXT_SETTINGS,
    Command,
    coro,
    device_options,
    emit,
    json_option,
    local_device_hint,
)


@click.command(cls=Command, context_settings=CONTEXT_SETTINGS)
@device_options
@json_option
@coro
async def doctor(device, config_path, platform, target, api_url, verbose, json_output):
    """Check system and device health; fail if any check fails."""
    # The check runner renders progress and a summary for Python callers.
    # The CLI emits its own single text/JSON result instead.
    with console.capture():
        results = await run_doctor(device, bool(verbose), config_path,
                                   target=target, platform=platform, api_url=api_url)
    failed = any(result.status == Status.FAIL for result in results)
    checks = [
        {"name": result.name, "status": result.status.value,
         "detail": local_device_hint("; ".join(filter(None, (result.message, result.detail))))}
        for result in results
    ]
    if json_output:
        emit({"ok": not failed, "checks": checks}, json_output=True)
    else:
        for check in checks:
            emit(f"{check['name']}: {check['status']}: {check['detail']}")
    if failed:
        click.get_current_context().exit(1)
