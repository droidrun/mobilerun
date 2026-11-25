"""Read-only Android diagnosis against a previously configured Portal forward."""

import subprocess
import sys

import pytest
from conftest import SERIAL


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
