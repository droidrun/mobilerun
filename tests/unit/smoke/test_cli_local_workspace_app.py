"""Exercise the offline local app-ID guard through the installed CLI."""

import subprocess
import sys


def test_local_workspace_id_fails_without_device_or_credentials(monkeypatch):
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mobilerun",
            "device",
            "app",
            "install",
            "--android",
            "--device",
            "offline-serial",
            "--app-id",
            "workspace-id",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "Pass a local APK path" in result.stderr
