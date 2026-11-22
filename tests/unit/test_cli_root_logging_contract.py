import importlib
import logging

from click.testing import CliRunner


def test_root_reuses_its_installed_handler(monkeypatch):
    app = importlib.import_module("mobilerun.cli.app")
    state = importlib.import_module("mobilerun.cli.base.logging")
    logger = logging.getLogger("mobilerun")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logger.level)
    monkeypatch.setattr(logger, "propagate", logger.propagate)
    monkeypatch.setattr(state, "_handler", None)
    runner = CliRunner()
    runner.invoke(app.cli, ["device", "--help"])
    handler = state._handler
    runner.invoke(app.cli, ["device", "--help"])
    assert state._handler is handler
    assert logger.handlers.count(handler) == 1
