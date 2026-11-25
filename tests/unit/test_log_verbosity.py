import logging

import pytest

from mobilerun.config.user_config import LoggingConfig
from mobilerun.logs.handlers import VERBOSE, configure_logging, resolve_log_level


@pytest.fixture(autouse=True)
def restore_logger():
    logger = logging.getLogger("mobilerun")
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    yield
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.mark.parametrize(
    ("cli", "env", "config", "debug", "expected"),
    [
        (None, None, "normal", False, "normal"),
        (None, None, "normal", True, "debug"),
        (None, "verbose", "quiet", True, "verbose"),
        ("quiet", "debug", "verbose", True, "quiet"),
    ],
)
def test_level_precedence(cli, env, config, debug, expected):
    assert (
        resolve_log_level(cli, env, LoggingConfig(level=config, debug=debug))
        == expected
    )


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("quiet", ["result"]),
        ("normal", ["goal", "step 1", "step 2", "result"]),
        ("verbose", ["goal", "step 1", "step 2", "reasoning", "result"]),
        ("debug", ["goal", "step 1", "step 2", "reasoning", "debug", "result"]),
    ],
)
def test_filtering(level, expected):
    class Capture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    handler = Capture()
    logger = logging.getLogger("mobilerun")
    old_handlers, old_level, old_propagate = (
        logger.handlers[:],
        logger.level,
        logger.propagate,
    )
    try:
        configure_logging(debug=False, handler=handler, level=level)
        logger.info("goal")
        logger.info("step 1")
        logger.info("step 2")
        logger.log(VERBOSE, "reasoning")
        logger.debug("debug")
        logger.warning("result")
        assert handler.messages == expected
    finally:
        logger.handlers = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def test_log_level_is_registered(monkeypatch):
    from mobilerun.config.settings import get_settings

    monkeypatch.setenv("MOBILERUN_LOG_LEVEL", "verbose")
    assert get_settings().MOBILERUN_LOG_LEVEL == "verbose"


def test_default_device_environment_setting_removed(monkeypatch):
    from mobilerun.config.settings import get_settings

    monkeypatch.setenv("MOBILERUN_DEFAULT_DEVICE", "obsolete")
    assert "default_device" not in get_settings().model_fields_set
    assert "default_device" not in type(get_settings()).model_fields
