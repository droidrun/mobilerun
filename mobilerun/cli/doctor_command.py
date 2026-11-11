"""Click wiring for health checks."""

from __future__ import annotations

import click

from mobilerun.cli.options import (
    Command,
    coro,
    device_options,
    emit,
    json_option,
    local_device_hint,
)


@click.command(cls=Command)
@device_options
@json_option
@coro
async def doctor(device, config_path, platform, target, api_url, verbose, json_output):
    """Check system and device health; fail if any check fails."""
    from mobilerun.cli.doctor import Status, run_doctor

    results = await run_doctor(
        device, bool(verbose), config_path, target=target, platform=platform,
        api_url=api_url, render_results=False,
    )
    failed = any(result.status == Status.FAIL for result in results)
    checks = [
        {"name": result.name, "status": result.status.value,
         "detail": local_device_hint("; ".join(filter(None, (result.message, result.detail))), target)}
        for result in results
    ]
    if json_output:
        emit({"ok": not failed, "checks": checks}, json_output=True)
    else:
        for check in checks:
            emit(f"{check['name']}: {check['status']}: {check['detail']}")
    if failed:
        click.get_current_context().exit(1)


from mobilerun.cli.options import _lazy_imports  # noqa: E402

__getattr__ = _lazy_imports(globals(), {
    'Status': ('mobilerun.cli.doctor', 'Status'),
    'run_doctor': ('mobilerun.cli.doctor', 'run_doctor'),
})

from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from mobilerun.cli.doctor import Status as Status
    from mobilerun.cli.doctor import run_doctor as run_doctor
