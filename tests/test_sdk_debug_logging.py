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

    (tmp_path / "android").mkdir()
    (tmp_path / "android" / "app_cards.json").write_text(
        json.dumps({"com.example": "a.md"})
    )
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

    assert "Loaded android/app_cards.json with 1 entries" in records.messages


def test_portal_setup_gets_mobilerun_version_when_supported():
    def new_setup(device, debug=False, version=None):
        return None

    def old_setup(device, debug=False):
        return None

    assert portal_version_kwargs(new_setup) == {"version": mobilerun.__version__}
    assert portal_version_kwargs(old_setup) == {}


def test_portal_setup_can_ask_for_an_in_place_update():
    def new_setup(device, debug=False, *, version=None, uninstall=True):
        return None

    def old_setup(device, debug=False):
        return None

    assert portal_version_kwargs(new_setup, in_place=True) == {
        "version": mobilerun.__version__,
        "uninstall": False,
    }
    assert portal_version_kwargs(new_setup) == {"version": mobilerun.__version__}
    assert portal_version_kwargs(old_setup, in_place=True) == {}


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


@pytest.mark.parametrize(
    "installed, status",
    [("0.7.26", "PASS"), ("0.7.26-dev", "PASS"), ("0.7.22", "WARN")],
)
def test_doctor_keeps_a_newer_portal_and_flags_an_older_one(
    monkeypatch, installed, status
):
    import asyncio

    from mobilerun.cli import doctor

    class Device:
        async def shell(self, command):
            return 'Row: 0 result={"status":"success","result":"' + installed + '"}'

    monkeypatch.setattr(doctor, "_get_latest_portal_version", lambda: "0.7.27")
    monkeypatch.setattr(
        doctor,
        "get_compatible_portal_version",
        lambda version, debug=False: ("0.7.25", "https://example.test", True),
    )

    result, _, _, _ = asyncio.run(doctor.check_portal_version(Device(), debug=False))

    assert result.status == doctor.Status[status]
    if installed.startswith("0.7.26"):
        assert "newer than 0.7.25" in result.message


@pytest.mark.parametrize("in_place, uninstall", [(True, False), (False, True)])
def test_setup_command_updates_in_place_only_when_asked(
    monkeypatch, in_place, uninstall
):
    import asyncio

    from mobilerun.cli import main

    calls = []

    async def fake_setup(device, debug=False, *, version=None, uninstall=True):
        calls.append({"version": version, "uninstall": uninstall})
        return True

    async def fake_device(serial):
        return object()

    monkeypatch.setattr(main, "setup_portal", fake_setup)
    monkeypatch.setattr(main.adb, "device", fake_device)

    asyncio.run(
        main._setup_portal(path=None, device="serial", debug=False, in_place=in_place)
    )

    assert calls[0]["version"] == mobilerun.__version__
    assert calls[0]["uninstall"] is uninstall


def test_sdk_version_check_still_flags_an_rc_after_its_release(monkeypatch):
    import asyncio

    from mobilerun.cli import doctor

    class Response:
        status_code = 200

        def json(self):
            return {"tag_name": "v0.6.22"}

    monkeypatch.setattr(doctor, "__version__", "0.6.22rc1")
    monkeypatch.setattr(doctor.requests, "get", lambda url, **kwargs: Response())

    result = asyncio.run(doctor.check_sdk_version(debug=False))

    assert result.status == doctor.Status.WARN
