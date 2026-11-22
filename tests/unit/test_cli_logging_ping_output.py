import importlib
import logging

import click
from click.testing import CliRunner


def test_root_debug_enables_debug_logging():
    from mobilerun.cli.app import cli

    logger = logging.getLogger("mobilerun")
    old_level = logger.level
    old_handlers = logger.handlers[:]
    old_propagate = logger.propagate
    try:
        result = CliRunner().invoke(cli, ["--debug", "device", "--help"])
        assert result.exit_code == 0
        assert logger.level == logging.DEBUG
    finally:
        logger.setLevel(old_level)
        logger.handlers = old_handlers
        logger.propagate = old_propagate


def test_cli_log_handler_writes_to_stderr():
    from mobilerun.logs.handlers import CLILogHandler

    @click.command()
    def command():
        handler = CLILogHandler()
        handler.emit(
            logging.LogRecord("mobilerun", logging.INFO, "", 1, "diagnostic", (), None)
        )

    result = CliRunner().invoke(command)
    assert result.stdout == ""
    assert "diagnostic" in result.stderr


def test_ping_missing_device_fails_on_stderr(monkeypatch):
    module = importlib.import_module("mobilerun.cli.devices.ping")

    async def missing(target):
        return None

    monkeypatch.setattr(module.adb, "device", missing)
    result = CliRunner().invoke(module.ping, ["missing-device"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "missing-device" in result.stderr


def test_ping_success_names_device(monkeypatch):
    module = importlib.import_module("mobilerun.cli.devices.ping")

    async def device(target):
        return object()

    async def ping(*args):
        pass

    monkeypatch.setattr(module.adb, "device", device)
    monkeypatch.setattr(module, "ping_portal", ping)
    monkeypatch.setattr(module, "ping_portal_content", ping)
    monkeypatch.setattr(module, "ping_portal_tcp", ping)
    result = CliRunner().invoke(module.ping, ["test-device"])
    assert result.exit_code == 0
    assert "✓" in result.stdout
    assert "test-device" in result.stdout
