"""Cloud CLI integration, with outputs kept inside the isolated temp directory."""

import subprocess
import sys
from pathlib import Path

import pytest
from conftest import CLOUD_ID

pytestmark = pytest.mark.smoke_cloud


@pytest.mark.parametrize(
    "args",
    [["device", "list", "--cloud"], ["auth", "status"]],
)
def test_cloud_cli(cloud_ready, args):
    result = subprocess.run(
        [str(Path(sys.executable).with_name("mobilerun")), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (
        result.returncode == 0
    ), "Cloud CLI failed (output withheld to protect credentials)"
    if args[:2] == ["device", "list"]:
        assert CLOUD_ID in result.stdout


def test_missing_cloud_key(real_env, monkeypatch):
    from mobilerun.core.devices.cloud import CloudDevice
    from mobilerun.errors import CloudAuthRequired

    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    with pytest.raises(CloudAuthRequired):
        CloudDevice(device_id=CLOUD_ID)


def test_cloud_ping_reports_unsupported(cloud_ready):
    result = subprocess.run(
        [str(Path(sys.executable).with_name("mobilerun")), "device", "ping", CLOUD_ID],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "supports Android targets only" in result.stderr
