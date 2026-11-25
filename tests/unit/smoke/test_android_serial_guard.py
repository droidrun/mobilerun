import os
import subprocess
import sys
from pathlib import Path

import pytest

from mobilerun.config.migrations import CURRENT_VERSION


@pytest.mark.smoke_device
def test_unavailable_android_serial_refuses_read_only_app_list(local_ready, tmp_path):
    """Even with an online phone, an unavailable serial cannot select it."""
    config = tmp_path / "device.yaml"
    config.write_text(f"_version: {CURRENT_VERSION}\ndevice: {{}}\n")
    serial = "nonexistent-mobilerun-smoke-serial"
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
        env={**os.environ, "ANDROID_SERIAL": serial},
    )
    assert result.returncode == 1 and f"ANDROID_SERIAL {serial}" in result.stderr
