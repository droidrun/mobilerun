"""Read-only smoke test: local Android has no clipboard endpoint."""

import subprocess
import sys
from pathlib import Path

import pytest
from conftest import SERIAL


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
    assert result.returncode == 1
    assert "AndroidDevice doesn't support get_clipboard" in result.stderr
