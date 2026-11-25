"""CLI default resolution against real devices; never invoke the introduction demo."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import CLOUD_ID, SERIAL


def cli(*args, env):
    return subprocess.run(
        [str(Path(sys.executable).with_name("mobilerun")), *args],
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
    )


@pytest.mark.smoke_device
def test_single_adb_device_default(local_ready):
    result = subprocess.run(
        ["adb", "devices"], capture_output=True, text=True, check=True
    )
    devices = [
        line.split()[0]
        for line in result.stdout.splitlines()[1:]
        if line.endswith("\tdevice")
    ]
    if devices != [SERIAL]:
        pytest.skip("requires exactly the configured local adb device")
    env = os.environ.copy()
    # This case exercises local-only discovery, not mixed local/cloud selection.
    env.pop("MOBILERUN_API_KEY", None)
    result = cli("device", "app", "list", env=env)
    assert result.returncode == 0, result.stderr
    assert f"Using {SERIAL}" in result.stderr
    assert result.stdout.strip()


@pytest.mark.smoke_device
def test_config_default(local_ready, tmp_path):
    from mobilerun.config.migrations import CURRENT_VERSION

    config = tmp_path / "device.yaml"
    config.write_text(f"_version: {CURRENT_VERSION}\ndevice:\n  serial: {SERIAL}\n")
    env = os.environ.copy()
    result = cli("device", "app", "list", "--config", str(config), env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


@pytest.mark.smoke_cloud
def test_cloud_config_default(cloud_ready, tmp_path):
    from mobilerun.config.migrations import CURRENT_VERSION

    config = tmp_path / "device.yaml"
    config.write_text(f"_version: {CURRENT_VERSION}\ndevice:\n  serial: {CLOUD_ID}\n")
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_path)
    result = cli("device", "screen", "screenshot", "--config", str(config), env=env)
    assert result.returncode == 0, result.stderr
    image = Path(result.stdout.strip())
    assert image.is_relative_to(tmp_path)
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
