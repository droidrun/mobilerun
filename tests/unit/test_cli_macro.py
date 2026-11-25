from click.testing import CliRunner
from test_macro_operator import FakeDevice, write_macro

from mobilerun.cli.devices.base import selection


def test_macro_group_exposes_list_and_replay():
    from mobilerun.macro.cli import macro

    result = CliRunner().invoke(macro, ["--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "replay" in result.output


def test_macro_replay_discovery_and_exclusive_flags(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices.base import options
    from mobilerun.macro.cli import macro

    monkeypatch.setattr(options, "discover_devices", AsyncMock(return_value=[]))
    path = write_macro(tmp_path)
    runner = CliRunner()
    missing = runner.invoke(macro, ["replay", str(path)])
    assert missing.exit_code == 1
    assert "No devices found." in missing.output
    multiple = runner.invoke(
        macro, ["replay", str(path), "target", "--android", "--ios"]
    )
    assert multiple.exit_code == 2
    assert "Choose at most one device type flag" in multiple.output


def fake_device_factory(monkeypatch):
    device = FakeDevice()
    targets = []

    async def create(target, **kwargs):
        targets.append(target)
        return device

    async def close():
        pass

    device.close = close

    monkeypatch.setattr(selection, "connect_target", create)
    return device, targets


def test_macro_cli_replays_on_selected_device(monkeypatch, tmp_path):
    from mobilerun.cli.app import cli

    device, targets = fake_device_factory(monkeypatch)
    result = CliRunner().invoke(
        cli, ["macro", "replay", str(write_macro(tmp_path)), "serial", "--delay", "0"]
    )
    assert result.exit_code == 0, result.output
    assert device.taps == [(10, 20)]
    assert targets == ["serial"]


def test_macro_cli_mismatch_exits_nonzero_without_action(monkeypatch, tmp_path):
    from mobilerun.cli.app import cli

    device, _ = fake_device_factory(monkeypatch)
    result = CliRunner().invoke(
        cli,
        [
            "macro",
            "replay",
            str(write_macro(tmp_path, True)),
            "serial",
            "--state-timeout",
            "0",
        ],
    )
    assert result.exit_code == 1
    assert "mismatch before step 1" in result.output
    assert device.taps == []


def test_dry_run_does_not_connect(monkeypatch, tmp_path):
    from mobilerun.cli.app import cli

    device, targets = fake_device_factory(monkeypatch)
    result = CliRunner().invoke(
        cli, ["macro", "replay", str(write_macro(tmp_path)), "serial", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "tap" in result.output
    assert targets == []
    assert device.taps == []
