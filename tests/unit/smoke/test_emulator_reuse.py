"""Free smoke test: reuse an existing emulator without stopping it."""

import asyncio
import subprocess

import pytest

from mobilerun.core.devices.android import AndroidDevice


@pytest.mark.smoke
def test_reuse_running_emulator():
    result = subprocess.run(
        ["adb", "-s", "emulator-5554", "emu", "avd", "name"],
        capture_output=True,
        text=True,
        check=True,
    )
    if not result.stdout.strip():
        result = subprocess.run(
            ["adb", "-s", "emulator-5554", "shell", "getprop", "ro.boot.qemu.avd_name"],
            capture_output=True,
            text=True,
            check=True,
        )
    name = result.stdout.splitlines()[0].strip()

    async def ready():
        return await AndroidDevice(avd=name)

    device = asyncio.run(ready())
    assert device._serial == "emulator-5554"
    assert not device._started_emulator
