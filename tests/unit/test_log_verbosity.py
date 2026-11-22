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


def test_quiet_cli_suppresses_banner_and_telemetry(monkeypatch, capsys):
    pytest.importorskip(
        "mobilerun.agents", reason="requires the optional mobilerun-agents add-on"
    )
    from mobilerun.agents.cli.agent.run import _show_local_config
    from mobilerun.agents.config.user_config import MobileConfig

    calls = []
    monkeypatch.setattr(
        "mobilerun.agents.cli.agent.run.print_telemetry_message",
        lambda **kw: calls.append(kw),
    )
    _show_local_config(MobileConfig(), False, "quiet")
    assert capsys.readouterr().out == ""
    assert calls == []


@pytest.mark.parametrize(
    ("env", "level", "debug", "expected"),
    [
        ("debug", None, None, True),
        ("debug", "quiet", None, False),
        ("quiet", None, True, True),
        ("debug", None, False, False),
    ],
)
def test_runner_passes_resolved_debug_to_agent(
    monkeypatch, env, level, debug, expected
):
    pytest.importorskip(
        "mobilerun.agents", reason="requires the optional mobilerun-agents add-on"
    )
    import asyncio
    from types import SimpleNamespace

    from mobilerun.agents import runner
    from mobilerun.agents.config.user_config import MobileConfig

    config = MobileConfig()
    config.credentials.enabled = False
    monkeypatch.setenv("MOBILERUN_LOG_LEVEL", env)
    monkeypatch.setattr(runner.ConfigLoader, "load", lambda path: config)
    observed = []

    class FakeHandler:
        async def stream_events(self):
            if False:
                yield None

        def __await__(self):
            async def result():
                return SimpleNamespace(success=True)

            return result().__await__()

    class FakeAgent:
        android_serial = None

        def __init__(self, *, config, **kwargs):
            observed.append((config.logging.level, config.logging.debug))

        def run(self):
            return FakeHandler()

    monkeypatch.setattr(runner, "MobileAgent", FakeAgent)
    result = asyncio.run(runner.run_agent("goal", log_level=level, debug=debug))
    assert result.success
    assert observed == [
        ("debug" if expected else "quiet" if level == "quiet" else "normal", expected)
    ]


def test_default_device_environment_setting_removed(monkeypatch):
    from mobilerun.config.settings import get_settings

    monkeypatch.setenv("MOBILERUN_DEFAULT_DEVICE", "obsolete")
    assert "default_device" not in get_settings().model_fields_set
    assert "default_device" not in type(get_settings()).model_fields


def test_cloud_applies_cli_level_before_device_run(monkeypatch):
    pytest.importorskip(
        "mobilerun.agents", reason="requires the optional mobilerun-agents add-on"
    )
    import asyncio
    from types import SimpleNamespace

    from mobilerun.agents import runner
    from mobilerun.agents.config.user_config import MobileConfig

    monkeypatch.setattr(runner.ConfigLoader, "load", lambda path=None: MobileConfig())
    monkeypatch.setenv("MOBILERUN_LOG_LEVEL", "debug")

    class Device:
        async def do(self, command, *, max_steps=None):
            assert logging.getLogger("mobilerun").level == logging.WARNING
            return SimpleNamespace(success=True)

    assert asyncio.run(
        runner.run_agent(
            "goal", device=Device(), device_kind="cloud", log_level="quiet"
        )
    ).success


def test_operator_applies_environment_level_to_agent(monkeypatch):
    pytest.importorskip(
        "mobilerun.agents", reason="requires the optional mobilerun-agents add-on"
    )
    import asyncio
    from types import SimpleNamespace

    from mobilerun.agents import operator
    from mobilerun.agents.config.user_config import MobileConfig

    monkeypatch.setenv("MOBILERUN_LOG_LEVEL", "debug")

    class Agent:
        trajectory = None

        def __init__(self, **kwargs):
            assert kwargs["config"].logging.debug is True
            assert kwargs["config"].logging.level == "debug"
            assert logging.getLogger("mobilerun").level == logging.DEBUG

        async def run(self):
            return SimpleNamespace(
                success=True, reason="done", structured_output=None, steps=1
            )

    monkeypatch.setattr(operator, "MobileAgent", Agent)
    assert asyncio.run(
        operator.AgentOperator(config=MobileConfig(), llms={}).execute(object(), "goal")
    ).success


def test_finalize_answer_is_distinct_from_reason(caplog, monkeypatch):
    pytest.importorskip(
        "mobilerun.agents", reason="requires the optional mobilerun-agents add-on"
    )
    from mobilerun.agents.agent.droid.events import FinalizeEvent
    from mobilerun.agents.logs.events import EventHandler

    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="mobilerun"):
        EventHandler().handle(
            FinalizeEvent(success=True, reason="finished", answer="done")
        )
    assert [record.message for record in caplog.records] == ["Done in 0 steps", "done"]


def test_legacy_finalize_event_has_no_answer(caplog, monkeypatch):
    pytest.importorskip(
        "mobilerun.agents", reason="requires the optional mobilerun-agents add-on"
    )
    from mobilerun.agents.agent.droid.events import FinalizeEvent
    from mobilerun.agents.logs.events import EventHandler

    event = FinalizeEvent(success=True, reason="finished")
    assert event.answer is None
    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="mobilerun"):
        EventHandler().handle(event)
    assert [record.message for record in caplog.records] == ["Done in 0 steps"]
