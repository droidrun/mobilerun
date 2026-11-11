"""Click wiring for health checks."""

from dataclasses import asdict

import click

from mobilerun.cli.doctor import Status, run_doctor
from mobilerun.cli.options import CONTEXT_SETTINGS, Command, coro, device_options, emit, json_option


@click.command(cls=Command, context_settings=CONTEXT_SETTINGS)
@device_options
@json_option
@coro
async def doctor(device, config_path, platform, target, api_url, verbose, json_output):
    """Check system and device health; fail if any check fails."""
    results = await run_doctor(device, bool(verbose), config_path,
                               target=target, platform=platform, api_url=api_url)
    failed = any(result.status == Status.FAIL for result in results)
    if not json_output:
        for result in results:
            emit(f"{result.name}: {result.status.value}: {result.message}")
    if failed:
        raise click.ClickException("Health checks failed")
    if json_output:
        emit([{**asdict(result), "status": result.status.value} for result in results], json_output=True)
