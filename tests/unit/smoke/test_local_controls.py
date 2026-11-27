"""Local Android smoke checks for discovery, controls, diagnostics and emulator reuse."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mobilerun.config.migrations import CURRENT_VERSION
from mobilerun.core.devices.android import AndroidDevice
from tests.support.smoke_settings import SERIAL


def _android_app_list(config, *, env=None):
    return subprocess.run(
        [
            str(Path(sys.executable).with_name("mobilerun")),
            "device",
            "app",
            "list",
            "--android",
            "--config",
            str(config),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
    )


@pytest.mark.smoke_device
def test_unavailable_android_serial_refuses_read_only_app_list(local_ready, tmp_path):
    """Even with an online phone, an unavailable serial cannot select it."""
    config = tmp_path / "device.yaml"
    config.write_text(f"_version: {CURRENT_VERSION}\ndevice: {{}}\n")
    serial = "nonexistent-mobilerun-smoke-serial"
    result = _android_app_list(config, env={**os.environ, "ANDROID_SERIAL": serial})
    assert result.returncode == 1 and f"ANDROID_SERIAL {serial}" in result.stderr


@pytest.mark.smoke_device
def test_grouped_app_list_discovers_only_android(local_ready, tmp_path):
    adb = subprocess.run(["adb", "devices"], check=True, capture_output=True, text=True)
    online = [
        line.split()[0] for line in adb.stdout.splitlines() if line.endswith("\tdevice")
    ]
    if online != [SERIAL]:
        pytest.skip("requires exactly the configured local Android device")
    config = tmp_path / "alternate.yaml"
    config.write_text(f"_version: {CURRENT_VERSION}\ndevice: {{}}\n")
    result = _android_app_list(config)
    assert result.returncode == 0, result.stderr
    assert f"Using {SERIAL}" in result.stderr
    assert result.stdout.strip()


@pytest.mark.smoke_device
def test_doctor_checks_existing_tcp_forward(local_ready):
    from mobilerun.config.settings import get_settings

    def portal_forwards():
        remote = f"tcp:{get_settings().MOBILERUN_PORTAL_REMOTE_PORT}"
        lines = subprocess.check_output(["adb", "forward", "--list"], text=True)
        return {
            line for line in lines.splitlines() if line.split()[::2] == [SERIAL, remote]
        }

    before = portal_forwards()
    if not before:
        pytest.skip("requires an existing Portal TCP forward")
    result = subprocess.run(
        [sys.executable, "-m", "mobilerun", "doctor", "-d", SERIAL],
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr
    assert portal_forwards() == before
    assert result.returncode == 0, output
    assert "State (tcp)" in output and "Screenshot (tcp)" in output


@pytest.mark.smoke_device
def test_local_android_clipboard_reports_unsupported(local_ready):
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("mobilerun")),
            "device",
            "input",
            "clipboard",
            "get",
            SERIAL,
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 2
    assert "Clipboard commands require a cloud device ID." in result.stderr


@pytest.mark.smoke_device
def test_local_device_actions_and_extensions(local_ready):
    executable = str(Path(sys.executable).with_name("mobilerun"))
    env = os.environ.copy()
    for command in (
        ["list", SERIAL],
        ["extensions", SERIAL, "--json"],
        ["call", "device.wait", SERIAL, "--arg", "seconds=0", "--json"],
    ):
        result = subprocess.run(
            [executable, "device", "action", *command],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        if command[0] == "list":
            assert "device.wait" in result.stdout
        else:
            json.loads(result.stdout)


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
