import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from mobilerun.telemetry import tracker
from mobilerun.telemetry.events import PackageVisitEvent

PRIMARY_TELEMETRY_ENV = "MOBILERUN_TELEMETRY_ENABLED"
LEGACY_TELEMETRY_ENV = "DROIDRUN_TELEMETRY_ENABLED"


def _package_visit_event() -> PackageVisitEvent:
    return PackageVisitEvent(
        package_name="com.example.settings",
        activity_name=".MainActivity",
        step_number=3,
    )


@pytest.fixture(autouse=True)
def reset_telemetry_state(monkeypatch):
    monkeypatch.delenv(PRIMARY_TELEMETRY_ENV, raising=False)
    monkeypatch.delenv(LEGACY_TELEMETRY_ENV, raising=False)
    monkeypatch.setattr(tracker, "_posthog", None)


def test_telemetry_is_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv(PRIMARY_TELEMETRY_ENV, raising=False)
    monkeypatch.delenv(LEGACY_TELEMETRY_ENV, raising=False)

    assert tracker.is_telemetry_enabled() is True
    assert tracker.is_telemetry_enabled(config_enabled=True) is True


def test_config_false_disables_telemetry() -> None:
    assert tracker.is_telemetry_enabled(config_enabled=False) is False


def test_environment_false_disables_telemetry(monkeypatch) -> None:
    monkeypatch.setenv(PRIMARY_TELEMETRY_ENV, "false")

    assert tracker.is_telemetry_enabled() is False
    assert tracker.is_telemetry_enabled(config_enabled=True) is False


def test_config_false_wins_when_environment_is_true(monkeypatch) -> None:
    monkeypatch.setenv(PRIMARY_TELEMETRY_ENV, "true")

    assert tracker.is_telemetry_enabled(config_enabled=False) is False


@pytest.mark.parametrize(
    ("primary", "legacy", "expected"),
    [
        ("false", "true", False),
        ("true", "false", True),
        ("", "true", False),
        (None, "false", False),
        (None, "yes", True),
    ],
)
def test_primary_environment_variable_takes_precedence_over_legacy(
    monkeypatch, primary: str | None, legacy: str, expected: bool
) -> None:
    if primary is None:
        monkeypatch.delenv(PRIMARY_TELEMETRY_ENV, raising=False)
    else:
        monkeypatch.setenv(PRIMARY_TELEMETRY_ENV, primary)
    monkeypatch.setenv(LEGACY_TELEMETRY_ENV, legacy)

    assert tracker.is_telemetry_enabled() is expected


def test_disabled_capture_does_not_resolve_user_or_create_client(monkeypatch) -> None:
    monkeypatch.setenv(PRIMARY_TELEMETRY_ENV, "true")
    calls = []

    def record_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("disabled telemetry performed a side effect")

    monkeypatch.setattr(tracker, "get_user_id", record_call)
    monkeypatch.setattr(tracker, "_get_posthog", record_call)

    tracker.capture(_package_visit_event(), config_enabled=False)

    assert calls == []
    assert tracker._posthog is None


def test_disabled_flush_does_not_flush_an_existing_client(monkeypatch) -> None:
    class FakePosthog:
        def __init__(self):
            self.flush_calls = 0

        def flush(self) -> None:
            self.flush_calls += 1

    client = FakePosthog()
    monkeypatch.setattr(tracker, "_posthog", client)
    monkeypatch.setenv(PRIMARY_TELEMETRY_ENV, "true")

    asyncio.run(tracker.flush(config_enabled=False))

    assert client.flush_calls == 0


def test_capture_remains_enabled_and_backward_compatible_by_default(
    monkeypatch,
) -> None:
    class FakePosthog:
        def __init__(self):
            self.captures = []

        def capture(self, event_name, *, distinct_id, properties) -> None:
            self.captures.append((event_name, distinct_id, properties))

    client = FakePosthog()
    monkeypatch.setattr(tracker, "_get_posthog", lambda: client)
    monkeypatch.setattr(tracker, "get_user_id", lambda: "generated-user-id")

    tracker.capture(_package_visit_event())
    tracker.capture(_package_visit_event(), "provided-user-id")

    assert [capture[0] for capture in client.captures] == [
        "PackageVisitEvent",
        "PackageVisitEvent",
    ]
    assert [capture[1] for capture in client.captures] == [
        "generated-user-id",
        "provided-user-id",
    ]
    assert all(capture[2]["run_id"] == tracker.RUN_ID for capture in client.captures)
    assert all(
        capture[2]["package_name"] == "com.example.settings"
        for capture in client.captures
    )


def test_posthog_client_is_constructed_once_under_concurrency(monkeypatch) -> None:
    workers = 16
    start_barrier = threading.Barrier(workers)
    constructed = []

    class FakePosthog:
        def __init__(self, **kwargs):
            # Release the GIL long enough for an unlocked lazy initializer to race.
            time.sleep(0.02)
            self.kwargs = kwargs
            constructed.append(self)

    monkeypatch.setattr(tracker, "Posthog", FakePosthog)

    def get_client():
        start_barrier.wait(timeout=5)
        return tracker._get_posthog()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        clients = list(executor.map(lambda _: get_client(), range(workers)))

    assert len(constructed) == 1
    assert all(client is constructed[0] for client in clients)
    assert constructed[0].kwargs == {
        "project_api_key": tracker.PROJECT_API_KEY,
        "host": tracker.HOST,
        "disable_geoip": False,
    }


def test_package_event_forwards_state_telemetry_setting(monkeypatch) -> None:
    from mobilerun.agent.droid import state as state_module

    captured = []

    def fake_capture(event, user_id=None, *, config_enabled=True) -> None:
        captured.append((event, user_id, config_enabled))

    monkeypatch.setattr(state_module, "capture", fake_capture)
    state = state_module.MobileAgentState(
        user_id="state-user",
        step_number=7,
        telemetry_config_enabled=False,
    )

    state.update_current_app("com.android.settings", ".Settings")

    assert len(captured) == 1
    event, user_id, config_enabled = captured[0]
    assert event == PackageVisitEvent(
        package_name="com.android.settings",
        activity_name=".Settings",
        step_number=7,
    )
    assert user_id == "state-user"
    assert config_enabled is False


def test_mobile_agent_copies_telemetry_config_to_shared_state(monkeypatch) -> None:
    from mobilerun.agent.droid import droid_agent as agent_module
    from mobilerun.config_manager.config_manager import (
        AgentConfig,
        MobileConfig,
        TelemetryConfig,
    )

    monkeypatch.setattr(agent_module, "setup_tracing", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        agent_module.MobileAgent,
        "_configure_default_logging",
        lambda *args, **kwargs: None,
    )
    config = MobileConfig(
        agent=AgentConfig(name="telemetry-test-external-agent"),
        telemetry=TelemetryConfig(enabled=False),
    )

    agent = agent_module.MobileAgent("Check telemetry", config=config)

    assert agent.shared_state.telemetry_config_enabled is False


def test_mobile_agent_lifecycle_forwards_disabled_telemetry_config(
    monkeypatch,
) -> None:
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.workflow import StartEvent

    from mobilerun.agent.droid import droid_agent as agent_module
    from mobilerun.agent.droid.events import FinalizeEvent
    from mobilerun.config_manager.config_manager import MobileConfig, TelemetryConfig
    from mobilerun.telemetry.events import (
        MobileAgentFinalizeEvent,
        MobileAgentInitEvent,
    )

    class FakeDriver:
        platform = "Android"
        supported = set()
        supported_buttons = set()

        async def get_date(self) -> str:
            return "2026-08-31"

    class FakeStateProvider:
        requires_coordinate_tools = False
        supported = set()

    class FakeRegistry:
        def disable_unsupported(self, capabilities) -> None:
            return None

        def disable(self, names) -> None:
            return None

        def register_from_dict(self, tools) -> None:
            return None

    class FakeContext:
        def __init__(self):
            self.events = []
            self.store = self

        def write_event_to_stream(self, event) -> None:
            self.events.append(event)

        async def set(self, key, value) -> None:
            setattr(self, key, value)

        async def get(self, key, default=None):
            return getattr(self, key, default)

    captures = []
    flushes = []

    def fake_capture(event, user_id=None, *, config_enabled=True) -> None:
        captures.append((event, user_id, config_enabled))

    async def fake_flush(*, config_enabled=True) -> None:
        flushes.append(config_enabled)

    async def fake_build_tool_registry(**kwargs):
        return FakeRegistry(), {"tap"}

    monkeypatch.setenv(PRIMARY_TELEMETRY_ENV, "true")
    monkeypatch.delenv("MOBILERUN_STREAM_SCREENSHOTS", raising=False)
    monkeypatch.delenv("DROIDRUN_STREAM_SCREENSHOTS", raising=False)
    monkeypatch.setattr(agent_module, "setup_tracing", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        agent_module.MobileAgent,
        "_configure_default_logging",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(agent_module, "build_tool_registry", fake_build_tool_registry)
    monkeypatch.setattr(agent_module, "capture", fake_capture)
    monkeypatch.setattr(agent_module, "flush", fake_flush)

    agent = agent_module.MobileAgent(
        "Check telemetry lifecycle",
        config=MobileConfig(telemetry=TelemetryConfig(enabled=False)),
        llms=MockLLM(),
        driver=FakeDriver(),
        state_provider=FakeStateProvider(),
        user_id="lifecycle-user",
    )
    context = FakeContext()

    async def run_lifecycle() -> None:
        await agent.start_handler(context, StartEvent())
        await agent.finalize(
            context,
            FinalizeEvent(success=True, reason="Lifecycle complete"),
        )

    asyncio.run(run_lifecycle())

    assert len(captures) == 2
    assert isinstance(captures[0][0], MobileAgentInitEvent)
    assert isinstance(captures[1][0], MobileAgentFinalizeEvent)
    assert [capture[1] for capture in captures] == [
        "lifecycle-user",
        "lifecycle-user",
    ]
    assert [capture[2] for capture in captures] == [False, False]
    assert flushes == [False]


def test_print_telemetry_message_respects_config_false(monkeypatch) -> None:
    messages = []
    monkeypatch.setenv(PRIMARY_TELEMETRY_ENV, "true")
    monkeypatch.setattr(tracker.mobilerun_logger, "debug", messages.append)

    tracker.print_telemetry_message(config_enabled=False)

    assert messages == [tracker.TELEMETRY_DISABLED_MESSAGE]


def test_cli_run_and_test_forward_loaded_telemetry_config(monkeypatch) -> None:
    from mobilerun.cli import main as cli_main
    from mobilerun.config_manager.config_manager import MobileConfig, TelemetryConfig

    class StopAfterTelemetryMessage(Exception):
        pass

    forwarded = []
    config = MobileConfig(telemetry=TelemetryConfig(enabled=False))

    def capture_config(*, config_enabled=True) -> None:
        forwarded.append(config_enabled)
        raise StopAfterTelemetryMessage

    async def skip_keyboard_cleanup(config, serial=None) -> None:
        return None

    monkeypatch.setattr(cli_main.ConfigLoader, "load", lambda _: config)
    monkeypatch.setattr(cli_main, "_setup_cli_logging", lambda _: None)
    monkeypatch.setattr(cli_main, "print_telemetry_message", capture_config)
    monkeypatch.setattr(cli_main, "_cleanup_android_keyboard", skip_keyboard_cleanup)
    monkeypatch.setattr(cli_main.console, "print", lambda *args, **kwargs: None)

    assert asyncio.run(cli_main.run_command("Check telemetry", debug=False)) is False
    assert asyncio.run(cli_main.test("Check telemetry", debug=False)) is None
    assert forwarded == [False, False]
