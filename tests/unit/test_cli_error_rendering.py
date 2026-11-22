import click
from click.testing import CliRunner

from mobilerun.cli.app import LazyGroup
from mobilerun.errors.base import DeviceError


class SampleGroup(LazyGroup):
    list_commands = click.Group.list_commands
    get_command = click.Group.get_command


@click.group(cls=SampleGroup)
@click.option("--debug", is_flag=True)
def sample(debug):
    pass


@sample.command()
def fail():
    raise DeviceError("bad\nrequest\x1b[31m")


@sample.command()
def bug():
    raise RuntimeError("internal secret")


@sample.command()
def interrupt():
    raise KeyboardInterrupt()


@sample.command()
def group_failure():
    first = DeviceError("one")
    first.add_note("device 0: Fake")
    raise ExceptionGroup("devices failed", [first, DeviceError("two")])


@sample.command()
def click_failure():
    raise click.ClickException("first\nsecond")


def test_expected_error_is_one_line():
    result = CliRunner().invoke(sample, ["fail"], color=True)
    assert result.exit_code == 1
    assert len(result.output.splitlines()) == 1
    assert "bad request" in result.output
    assert "\x1b" not in result.output


def test_unexpected_error_hides_details_unless_debug():
    result = CliRunner().invoke(sample, ["bug"])
    assert result.exit_code == 1
    assert "internal secret" not in result.output
    assert "--debug" in result.output
    debug = CliRunner().invoke(sample, ["--debug", "bug"])
    assert debug.exit_code == 1
    assert "Traceback" in debug.output and "internal secret" in debug.output


def test_interrupt_and_usage_codes():
    runner = CliRunner()
    assert runner.invoke(sample, ["interrupt"]).exit_code == 130
    assert runner.invoke(sample, ["missing"]).exit_code == 2


def test_group_reports_each_cause():
    result = CliRunner().invoke(sample, ["group-failure"])
    assert result.exit_code == 1
    assert len(result.output.splitlines()) == 3
    assert "device 0" in result.output


def test_click_failure_is_one_line():
    result = CliRunner().invoke(sample, ["click-failure"])
    assert result.exit_code == 1
    assert result.output.splitlines() == ["Error: first second"]
