"""Free local CLI coverage for action discovery and invocation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import SERIAL


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
