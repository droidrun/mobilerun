"""Read status on the existing smoke phone without printing its response."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import CLOUD_ID


@pytest.mark.smoke_cloud
def test_status_json_excludes_connection_credentials(cloud_ready):
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("mobilerun")),
            "device",
            "cloud",
            "status",
            CLOUD_ID,
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        pytest.fail("Cloud status command failed; output withheld")
    fields = json.loads(result.stdout).keys()
    assert {"id", "state"} <= fields
    assert not {"streamToken", "stream_token", "streamUrl", "stream_url"}.intersection(
        fields
    )
