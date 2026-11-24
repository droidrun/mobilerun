"""Read-only number capacity CLI smoke test; creates no resources."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.smoke_cloud
def test_number_included_all_countries(real_env):
    if not os.getenv("MOBILERUN_API_KEY"):
        pytest.skip("Cloud API key is not set")
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("mobilerun")),
            "number",
            "included",
            "--billing",
            "no_commerce",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, "Cloud CLI failed (output withheld)"
    entries = json.loads(result.stdout)
    assert isinstance(entries, list)
    assert all(set(entry) == {"country", "remaining"} for entry in entries)
    assert all(
        isinstance(entry["country"], str)
        and (entry["remaining"] is None or isinstance(entry["remaining"], int))
        for entry in entries
    )
