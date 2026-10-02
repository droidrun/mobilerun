import json
import logging

import pytest

import mobilerun  # noqa: F401  (attaches the import-time handler)
from mobilerun.agent.droid import droid_agent as agent_module
from mobilerun.agent.utils.portal_setup import portal_version_kwargs


@pytest.fixture
def mobilerun_logger():
    log = logging.getLogger("mobilerun")
    saved = (list(log.handlers), log.level, log.propagate)
    yield log
    log.handlers, log.level, log.propagate = saved[0], saved[1], saved[2]


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_debug_flag_raises_the_level_with_the_import_time_handler(mobilerun_logger):
    assert mobilerun_logger.handlers
    mobilerun_logger.setLevel(logging.INFO)

    agent_module.MobileAgent._configure_default_logging(debug=True)

    assert mobilerun_logger.level == logging.DEBUG


def test_debug_off_keeps_a_level_the_user_set(mobilerun_logger):
    mobilerun_logger.setLevel(logging.DEBUG)

    agent_module.MobileAgent._configure_default_logging(debug=False)

    assert mobilerun_logger.level == logging.DEBUG


def test_sdk_debug_config_logs_app_card_loading(
    mobilerun_logger, monkeypatch, tmp_path
):
    from llama_index.core.llms.mock import MockLLM

    from mobilerun.config_manager.config_manager import (
        AgentConfig,
        AppCardConfig,
        LoggingConfig,
        MobileConfig,
        TelemetryConfig,
    )

    (tmp_path / "app_cards.json").write_text(json.dumps({"com.example": "a.md"}))
    mobilerun_logger.setLevel(logging.INFO)
    records = _Records()
    mobilerun_logger.addHandler(records)
    monkeypatch.setattr(agent_module, "setup_tracing", lambda *a, **kw: None)

    agent_module.MobileAgent(
        "Open the app",
        config=MobileConfig(
            agent=AgentConfig(
                reasoning=True, app_cards=AppCardConfig(app_cards_dir=str(tmp_path))
            ),
            logging=LoggingConfig(debug=True),
            telemetry=TelemetryConfig(enabled=False),
        ),
        llms=MockLLM(),
    )

    assert "Loaded app_cards.json with 1 entries" in records.messages


def test_portal_setup_gets_mobilerun_version_when_supported():
    def new_setup(device, debug=False, version=None):
        return None

    def old_setup(device, debug=False):
        return None

    assert portal_version_kwargs(new_setup) == {"version": mobilerun.__version__}
    assert portal_version_kwargs(old_setup) == {}


def test_doctor_passes_when_the_pinned_portal_is_installed(monkeypatch):
    import asyncio

    from mobilerun.cli import doctor

    class Device:
        async def shell(self, command):
            return 'Row: 0 result={"status":"success","result":"0.7.25"}'

    monkeypatch.setattr(doctor, "_get_latest_portal_version", lambda: "0.7.26")
    monkeypatch.setattr(
        doctor,
        "get_compatible_portal_version",
        lambda version, debug=False: ("0.7.25", "https://example.test", True),
    )

    result, installed, expected, _ = asyncio.run(
        doctor.check_portal_version(Device(), debug=False)
    )

    assert result.status == doctor.Status.PASS
    assert (installed, expected) == ("0.7.25", "0.7.25")
