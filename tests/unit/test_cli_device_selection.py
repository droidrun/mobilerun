import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from click.testing import CliRunner

from mobilerun.cli.devices.base.discovery import DiscoveredDevice


def test_device_picker_precedes_coordinates(monkeypatch):
    from mobilerun.cli.base import menu
    from mobilerun.cli.devices import device
    from mobilerun.cli.devices.base import options
    from mobilerun.cli.devices.base import selection as tap

    order = []
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(
            return_value=[
                DiscoveredDevice("a", "local", None, None, None),
                DiscoveredDevice("b", "local", None, None, None),
            ]
        ),
    )
    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    def pick(*args):
        order.append("device")
        return "a"

    def text(label, **kwargs):
        order.append(label)
        return "1"

    monkeypatch.setattr(options, "select_prompt", pick)
    monkeypatch.setattr(menu, "text_prompt", text)
    monkeypatch.setattr(options, "confirm_prompt", lambda *args, **kwargs: False)
    monkeypatch.setattr(tap, "connect_target", AsyncMock(return_value=AsyncMock()))
    result = CliRunner().invoke(device, ["tap"])
    assert result.exit_code == 0, result.output
    assert order == ["device", "X coordinate", "Y coordinate"]


def test_list_marks_default(monkeypatch):
    from mobilerun.cli.devices import list as listing

    monkeypatch.setattr(
        listing,
        "discover_devices",
        AsyncMock(
            return_value=[
                DiscoveredDevice("a", "local", None, None, None),
                DiscoveredDevice("b", "cloud", "Phone", "ready", "android"),
            ]
        ),
    )
    monkeypatch.setattr(
        listing.ConfigLoader,
        "load",
        lambda *args: SimpleNamespace(device=SimpleNamespace(serial="b")),
    )
    runner = CliRunner()
    result = runner.invoke(listing.devices, ["--json"])
    assert [item["default"] for item in json.loads(result.stdout)] == [False, True]
    result = runner.invoke(listing.devices)
    assert "★" in result.stdout and "b" in result.stdout


def test_unavailable_saved_default_errors_without_fallback(monkeypatch):
    import click

    from mobilerun.cli.devices.base import options

    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial="missing")),
    )
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(return_value=[DiscoveredDevice("other", "local", None, None, None)]),
    )
    monkeypatch.setattr(options, "is_interactive", lambda: False)

    @click.command()
    @options.device_options
    def command(target, **kwargs):
        click.echo(target)

    result = CliRunner().invoke(command)
    assert result.exit_code == 1
    assert "Default device missing is not available." in result.output


def test_picker_save_yes_and_no(monkeypatch):
    import click

    from mobilerun.cli.devices.base import options

    devices = [
        DiscoveredDevice("a", "local", None, None, None),
        DiscoveredDevice("b", "local", None, None, None),
    ]
    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial=None)),
    )
    monkeypatch.setattr(options, "discover_devices", AsyncMock(return_value=devices))
    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(options, "select_prompt", lambda *args: "a")
    saved = []
    monkeypatch.setattr(options, "save_default_device", saved.append)

    @click.command()
    @options.device_options
    def command(target, **kwargs):
        click.echo(target)

    for answer in (False, True):
        monkeypatch.setattr(
            options, "confirm_prompt", lambda *args, answer=answer, **kwargs: answer
        )
        result = CliRunner().invoke(command)
        assert result.exit_code == 0, result.output
    assert saved == ["a"]


def test_existing_default_no_save_prompt(monkeypatch):
    import click

    from mobilerun.cli.devices.base import options

    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(device=SimpleNamespace(serial="a")),
    )
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(return_value=[DiscoveredDevice("a", "local", None, None, None)]),
    )
    monkeypatch.setattr(options, "is_interactive", lambda: True)

    def fail(*args, **kwargs):
        raise AssertionError("unexpected confirmation")

    monkeypatch.setattr(options, "confirm_prompt", fail)

    @click.command()
    @options.device_options
    def command(target, **kwargs):
        click.echo(target)

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert result.stdout == "a\n"


def test_explicit_device_option_precedes_prompt_without_discovery(monkeypatch):
    from mobilerun.cli.base import menu
    from mobilerun.cli.devices import device
    from mobilerun.cli.devices.base import options
    from mobilerun.cli.devices.base import selection as tap

    monkeypatch.setattr(options, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(
        options,
        "discover_devices",
        AsyncMock(side_effect=AssertionError("unexpected discovery")),
    )
    monkeypatch.setattr(menu, "text_prompt", lambda *args: "1")
    driver = AsyncMock()
    connect = AsyncMock(return_value=driver)
    monkeypatch.setattr(tap, "connect_target", connect)
    result = CliRunner().invoke(device, ["tap", "--device", "chosen"])
    assert result.exit_code == 0, result.output
    assert connect.await_args.args[0] == "chosen"
