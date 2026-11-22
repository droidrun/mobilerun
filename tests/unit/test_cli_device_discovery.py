import asyncio
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner


def test_discovery_local_and_cloud(monkeypatch):
    from mobilerun.cli.devices.base import discovery

    async def local():
        return [SimpleNamespace(serial="usb-1")]

    async def cloud(**kwargs):
        return [
            SimpleNamespace(id="cloud-1", name="Phone", state="stopped", type="android")
        ]

    monkeypatch.setattr(discovery.adb, "list", local)
    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: "key")
    monkeypatch.setattr(discovery, "list_devices", cloud)
    devices = asyncio.run(discovery.discover_devices())
    assert [(d.id, d.source, d.ready) for d in devices] == [
        ("usb-1", "local", True),
        ("cloud-1", "cloud", False),
    ]


@pytest.mark.parametrize("interactive", [True, False])
@pytest.mark.parametrize("count", [0, 1, 2])
def test_selection_count(monkeypatch, interactive, count):
    from mobilerun.cli.devices.base import options
    from mobilerun.cli.devices.base.discovery import DiscoveredDevice

    devices = [
        DiscoveredDevice(str(i), "local", None, None, None) for i in range(count)
    ]

    async def discover(**kwargs):
        return devices

    monkeypatch.setattr(options, "discover_devices", discover)
    monkeypatch.setattr(options, "is_interactive", lambda: interactive)
    monkeypatch.setattr(options, "select_prompt", lambda *args: "1")
    monkeypatch.setattr(options, "confirm_prompt", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial=None)),
    )

    @click.command()
    @options.device_options
    def command(target, **kwargs):
        click.echo(target)

    result = CliRunner().invoke(command, [])
    if count == 0:
        assert result.exit_code == 1
        assert "No devices found." in result.output
    elif count == 1:
        assert result.exit_code == 0
        assert result.stdout == "0\n"
        assert "Using 0" in result.stderr
    elif interactive:
        assert result.exit_code == 0
        assert result.stdout == "1\n"
    else:
        assert result.exit_code == 1
        assert "0 (local)" in result.output and "1 (local)" in result.output


def test_missing_key_skips_cloud(monkeypatch):
    from mobilerun.cli.devices.base import discovery

    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: None)
    assert asyncio.run(discovery.discover_devices(local=False)) == []


def test_cloud_error_warns_and_returns_local(monkeypatch, capsys):
    from mobilerun.cli.devices.base import discovery

    async def local():
        return [SimpleNamespace(serial="usb")]

    async def cloud(**kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(discovery.adb, "list", local)
    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: "key")
    monkeypatch.setattr(discovery, "list_devices", cloud)
    assert [d.id for d in asyncio.run(discovery.discover_devices())] == ["usb"]
    output = capsys.readouterr().err
    assert "--debug" in output and "offline" not in output


@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_and_default_skip_discovery(monkeypatch, explicit):
    from mobilerun.cli.devices.base import options

    async def fail(**kwargs):
        if explicit:
            raise AssertionError("discovery called")
        from mobilerun.cli.devices.base.discovery import DiscoveredDevice

        return [DiscoveredDevice("default", "local", None, None, None)]

    monkeypatch.setattr(options, "discover_devices", fail)
    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial="default")),
    )

    @click.command()
    @options.device_options
    def command(target, **kwargs):
        click.echo(target)

    result = CliRunner().invoke(command, ["chosen"] if explicit else [])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == ("chosen" if explicit else "default")


def test_picker_labels_not_ready(monkeypatch):
    from mobilerun.cli.devices.base import options
    from mobilerun.cli.devices.base.discovery import DiscoveredDevice

    async def discover(**kwargs):
        return [
            DiscoveredDevice("usb", "local", None, None, None),
            DiscoveredDevice("cloud", "cloud", "Phone", "stopped", "android"),
        ]

    labels = []

    def select(message, choices):
        labels.extend(choice.label for choice in choices)
        return "cloud"

    monkeypatch.setattr(options, "discover_devices", discover)
    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(options, "select_prompt", select)
    monkeypatch.setattr(options, "confirm_prompt", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial=None)),
    )

    @click.command()
    @options.device_options
    def command(target, **kwargs):
        click.echo(target)

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert "cloud (Phone) [stopped] cloud (not ready)" in labels


def test_cancel_and_empty_picker(monkeypatch):
    from mobilerun.cli.devices.base import options
    from mobilerun.cli.devices.base.discovery import DiscoveredDevice

    async def discover(**kwargs):
        return [
            DiscoveredDevice("a", "local", None, None, None),
            DiscoveredDevice("b", "local", None, None, None),
        ]

    monkeypatch.setattr(options, "discover_devices", discover)
    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial=None)),
    )

    @click.command()
    @options.device_options
    def command(target, **kwargs):
        click.echo(target)

    monkeypatch.setattr(options, "select_prompt", lambda *args: None)
    assert CliRunner().invoke(command).exit_code == 0

    def cancel(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(options, "select_prompt", cancel)
    assert CliRunner().invoke(command).exit_code == 130


def test_device_option_wins_over_default(monkeypatch):
    from mobilerun.cli.devices.base import options

    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial=None)),
    )

    def command(target, **kwargs):
        click.echo(target)

    command.__module__ = "mobilerun.cli.devices.test"
    wrapped = click.command()(options.device_options(command))
    result = CliRunner().invoke(wrapped, ["--device", "chosen"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "chosen\n"
