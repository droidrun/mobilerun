"""Read-only discovery through the grouped app command on a local phone."""

import subprocess
import sys
from pathlib import Path

import pytest
from conftest import SERIAL


@pytest.mark.smoke_device
def test_grouped_app_list_discovers_only_android(local_ready, tmp_path):
    from mobilerun.config.migrations import CURRENT_VERSION

    adb = subprocess.run(["adb", "devices"], check=True, capture_output=True, text=True)
    online = [
        line.split()[0] for line in adb.stdout.splitlines() if line.endswith("\tdevice")
    ]
    if online != [SERIAL]:
        pytest.skip("requires exactly the configured local Android device")
    config = tmp_path / "alternate.yaml"
    config.write_text(f"_version: {CURRENT_VERSION}\ndevice: {{}}\n")
    result = subprocess.run(
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
    )
    assert result.returncode == 0, result.stderr
    assert f"Using {SERIAL}" in result.stderr
    assert result.stdout.strip()
