import asyncio

import pytest
from llama_index.core.workflow import StartEvent
from pydantic import BaseModel

import mobilerun.agent.droid.droid_agent as droid_module
from mobilerun.agent.utils import llm_loader
from mobilerun.config_manager import env_keys
from mobilerun.config_manager.config_manager import (
    AgentConfig,
    MobileConfig,
    TelemetryConfig,
)

KEY = "tsk_test_jevmobile_3Fd8Kq1Wz6Lp9Rb2"
GOAL = "Read the battery level from Settings"


class Battery(BaseModel):
    level_percent: int


class FakeJev:
    model = "jev-latest"

    def __init__(self):
        self.list_models_calls = 0
        self.closed = False

    async def list_models(self):
        self.list_models_calls += 1
        return ["jev-latest"]

    async def aclose(self):
        self.closed = True


class FakeContext:
    def __init__(self):
        self.events = []
        self.store = self

    def write_event_to_stream(self, event):
        self.events.append(event)

    async def set(self, key, value):
        setattr(self, key, value)

    async def get(self, key, default=None):
        return getattr(self, key, default)


@pytest.fixture(autouse=True)
def mobile_env(monkeypatch, tmp_path):
    """Quiet MobileAgent setup; any LLM load fails the test."""

    def no_llm(*args, **kwargs):
        raise AssertionError("Jev must not load an LLM")

    async def no_flush(**kwargs):
        return None

    monkeypatch.setattr(env_keys, "AUTH_PROFILES_PATH", tmp_path / "auth-profiles.json")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("MOBILERUN_TELEMETRY_ENABLED", "false")
    monkeypatch.setattr(droid_module, "setup_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(
        droid_module.MobileAgent, "_configure_default_logging", lambda *a, **kw: None
    )
    monkeypatch.setattr(droid_module, "capture", lambda *a, **kw: None)
    monkeypatch.setattr(droid_module, "flush", no_flush)
    monkeypatch.setattr(llm_loader, "load_llm", no_llm)


@pytest.fixture
def clients(monkeypatch):
    created = []

    def make_client(api_key, **kwargs):
        client = FakeJev()
        created.append(client)
        return client

    monkeypatch.setattr(droid_module, "JevClient", make_client)
    return created


def _jev_config():
    return MobileConfig(
        agent=AgentConfig(name="jev", after_sleep_action=0.0),
        telemetry=TelemetryConfig(enabled=False),
    )


@pytest.mark.parametrize("with_key", [True, False])
def test_output_model_is_rejected_before_any_setup(monkeypatch, clients, with_key):
    if with_key:
        monkeypatch.setenv("TYPESAFE_API_KEY", KEY)

    with pytest.raises(
        ValueError, match="output_model is not supported with the Jev agent"
    ):
        droid_module.MobileAgent(GOAL, config=_jev_config(), output_model=Battery)

    assert clients == []


def test_jev_requires_no_structured_output_profile():
    config = _jev_config()

    assert llm_loader._get_required_profiles(config, output_model=Battery) == []


def test_client_is_closed_when_device_setup_fails(monkeypatch, clients):
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    serial_calls = []

    async def no_device():
        serial_calls.append(1)
        raise RuntimeError("no devices connected")

    monkeypatch.setattr(droid_module, "resolve_android_serial", no_device)
    agent = droid_module.MobileAgent(GOAL, config=_jev_config())

    for attempt in (1, 2):
        with pytest.raises(RuntimeError, match="no devices connected"):
            asyncio.run(agent.start_handler(FakeContext(), StartEvent()))

        assert len(serial_calls) == attempt
        assert agent.jev_client is None
        assert len(clients) == attempt
        assert clients[-1].list_models_calls == 1
        assert clients[-1].closed is True
