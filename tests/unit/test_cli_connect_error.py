from unittest.mock import AsyncMock

from click.testing import CliRunner

from mobilerun.cli.devices.connect import connect


def test_connect_failure_exits_nonzero(monkeypatch):
    monkeypatch.setattr(
        "mobilerun.cli.devices.connect.adb.connect", AsyncMock(return_value="failed")
    )
    result = CliRunner().invoke(connect, ["localhost:5555"])
    assert result.exit_code == 1
