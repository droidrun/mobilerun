"""Click wiring for health checks."""

from dataclasses import asdict

import click

from mobilerun.cli.device_commands import _teardown_android, connect_driver
from mobilerun.cli.doctor import CheckResult, Status, check_config, check_sdk_version, run_doctor
from mobilerun.cli.options import CONTEXT_SETTINGS, Command, coro, device_options, emit, json_option
from mobilerun.cli.selection import select_device


@click.command(cls=Command, context_settings=CONTEXT_SETTINGS)
@device_options
@json_option
@coro
async def doctor(device, config_path, tcp, platform, target, api_url, verbose, json_output):
    """Check system and device health; fail if any check fails."""
    try:
        selection = await select_device(device, config_path, tcp, platform, target)
    except Exception as exc:
        results = [CheckResult("Device", Status.FAIL, str(exc))]
    else:
        if selection.backend == "local-android-adb":
            results = await run_doctor(selection.device, bool(verbose), config_path)
        else:
            config_result, _ = await check_config(bool(verbose), config_path)
            results = [await check_sdk_version(bool(verbose)), config_result]
            try:
                driver, _ = await connect_driver(
                    device, config_path, tcp, platform, target, api_url, verbose
                )
                try:
                    await driver.get_ui_tree()
                finally:
                    await _teardown_android(driver)
                results.append(CheckResult("Device", Status.PASS, "UI tree is reachable"))
            except Exception as exc:
                results.append(CheckResult("Device", Status.FAIL, str(exc)))
    if json_output:
        emit(
            [{**asdict(result), "status": result.status.value} for result in results],
            json_output=True,
        )
    else:
        for result in results:
            emit(f"{result.name}: {result.status.value}: {result.message}")
    if any(result.status == Status.FAIL for result in results):
        raise click.ClickException("Health checks failed")
