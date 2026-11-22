from click.testing import CliRunner


def test_device_options_expose_positional_target_and_type_flags():
    import click

    from mobilerun.cli.devices.base.options import device_options

    @click.command()
    @device_options
    def command(**kwargs):
        pass

    result = CliRunner().invoke(command, ["--help"])
    assert result.exit_code == 0
    assert "--cloud" in result.output
    assert "DEVICE" in result.output
    assert "-a, --android" in result.output
    assert "-i, --ios" in result.output
    assert "-v, --remote" in result.output
    assert "-c, --cloud" in result.output


def test_device_type_flags_force_detected_kind_and_reject_multiple():
    import click
    import pytest

    from mobilerun.cli.devices.base.options import selected_device_kind

    assert (
        selected_device_kind("http://phone", False, True, False, False, None, None)
        == "ios"
    )
    assert (
        selected_device_kind("not-a-uuid", False, False, True, False, None, None)
        == "cloud"
    )
    assert (
        selected_device_kind("serial", True, False, False, False, None, None)
        == "android"
    )
    assert (
        selected_device_kind("http://remote", False, False, False, True, None, None)
        == "visual_remote"
    )
    with pytest.raises(click.UsageError, match="at most one"):
        selected_device_kind("serial", True, True, False, False, None, None)


def test_device_target_precedence(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import click

    from mobilerun.cli.devices.base import options

    config = SimpleNamespace(device=SimpleNamespace(serial="saved"))
    monkeypatch.setattr(options.ConfigLoader, "load", lambda path: config)
    from mobilerun.cli.devices.base.discovery import DiscoveredDevice

    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(
            return_value=[
                DiscoveredDevice("discovered", "local", None, None, None),
                DiscoveredDevice("saved", "local", None, None, None),
            ]
        ),
    )
    monkeypatch.setenv("MOBILERUN_DEFAULT_DEVICE", "environment")

    def command(target, **kwargs):
        click.echo(target)

    command.__module__ = "mobilerun.cli.devices.test"
    command = click.command()(options.device_options(command))
    runner = CliRunner()
    explicit_config = tmp_path / "config.yaml"
    explicit_config.touch()
    assert (
        "explicit"
        in runner.invoke(command, ["explicit", "--config", str(explicit_config)]).output
    )
    assert "saved" in runner.invoke(command, ["--config", str(explicit_config)]).output
    assert "saved" in runner.invoke(command, []).output
    monkeypatch.delenv("MOBILERUN_DEFAULT_DEVICE")
    assert "saved" in runner.invoke(command, []).output
    config.device.serial = None
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(
            return_value=[DiscoveredDevice("discovered", "local", None, None, None)]
        ),
    )
    assert "discovered" in runner.invoke(command, []).output
