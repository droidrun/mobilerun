import asyncio
import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from mobilerun_core_local.driver.ios import IOSPortalDriver, IOSPortalHttpDriver

from mobilerun.config_manager import env_keys
from mobilerun.config_manager.config_manager import (
    AgentConfig,
    MobileConfig,
    TelemetryConfig,
)
from mobilerun.config_manager.loader import ConfigLoader
from mobilerun.macro import cli as macro_cli_module
from mobilerun.macro import replay as replay_module
from mobilerun.macro.handoff import run_agent_handoff
from mobilerun.macro.replay import MacroPlayer
from mobilerun.macro.state import normalize_ui_state
from mobilerun.tools.filters import DetailedFilter
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.ios_provider import IOSStateProvider
from mobilerun.tools.ui.provider import AndroidStateProvider

PORTAL_URL = "http://127.0.0.1:8080"

LOCAL_PORTAL_STATE = {
    "a11y_tree": {
        "boundsInScreen": {"left": 0, "top": 0, "right": 393, "bottom": 852},
        "className": "Application",
        "resourceId": "",
        "text": "",
        "contentDescription": "",
        "packageName": "com.apple.Preferences",
        "children": [
            {
                "boundsInScreen": {"left": 16, "top": 180, "right": 377, "bottom": 224},
                "className": "Button",
                "resourceId": "Privacy",
                "text": "Privacy & Security",
                "contentDescription": "",
                "packageName": "com.apple.Preferences",
                "children": [],
            }
        ],
    },
    "phone_state": {"currentApp": "Settings", "packageName": "com.apple.Preferences"},
    "device_context": {"screen_bounds": {"width": 393, "height": 852}},
}

LEGACY_A11Y = """Attributes: Application, pid: 123, label: 'Settings'
Element subtree:
 ->Application, pid: 123, label: 'Settings'
    Window (Main), {{0.0, 0.0}, {440.0, 956.0}}
      Cell, {{20.0, 200.0}, {400.0, 44.0}}, label: 'General'
"""


class FakeAndroidDriver:
    platform = "android"

    def __init__(self, state):
        self.state = state
        self.taps = []

    async def connect(self):
        return None

    async def get_ui_tree(self):
        return self.state

    async def tap(self, x, y):
        self.taps.append((x, y))


class FakePortalHttpDriver(IOSPortalHttpDriver):
    def __init__(self, state=LOCAL_PORTAL_STATE):  # skip the httpx client setup
        self.state = state
        self.connected = False
        self.taps = []

    async def connect(self):
        self.connected = True

    async def get_ui_tree(self):
        return self.state

    async def tap(self, x, y):
        self.taps.append((x, y))


class FakeLegacyIOSDriver(IOSPortalDriver):
    def __init__(self):  # skip the httpx client setup
        pass

    async def get_ui_tree(self):
        return {
            "a11y_tree": LEGACY_A11Y,
            "phone_state": {"currentApp": "Settings"},
            "device_context": {"screen_bounds": {"width": 440, "height": 956}},
        }


def _android_state(text="Continue"):
    return {
        "a11y_tree": [
            {
                "resourceId": "com.example:id/continue",
                "className": "android.widget.Button",
                "text": text,
                "clickable": True,
                "enabled": True,
                "bounds": "10,20,110,60",
            }
        ],
        "phone_state": {"package": "com.example", "activity": ".MainActivity"},
    }


def _macro(pre_state, description="finish task"):
    return {
        "macro_schema_version": "2.0",
        "version": "2.0",
        "description": description,
        "total_actions": 1,
        "actions": [{"action_type": "tap", "x": 60, "y": 40, "pre_state": pre_state}],
    }


def _portal_snapshot(state=LOCAL_PORTAL_STATE):
    provider = AndroidStateProvider(
        FakePortalHttpDriver(state),
        tree_filter=DetailedFilter(),
        tree_formatter=IndexedFormatter(),
    )
    return normalize_ui_state(asyncio.run(provider.get_state()))


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ConfigLoader, "get_user_config_path", classmethod(lambda cls: tmp_path / "x")
    )
    monkeypatch.delenv("MOBILERUN_DEVICE_TOKEN", raising=False)
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)


@pytest.fixture
def fake_ios_portal(monkeypatch, isolated_config):
    created = {}

    async def fake_discover():
        created["discovered"] = True
        return PORTAL_URL

    async def fake_create_ios_driver(url, *, token=None, transport=None):
        created["url"] = url
        created["token"] = token
        created["driver"] = FakePortalHttpDriver()
        return created["driver"]

    monkeypatch.setattr(replay_module, "discover_ios_device", fake_discover)
    monkeypatch.setattr(replay_module, "create_ios_driver", fake_create_ios_driver)
    return created


def test_agent_mode_forwards_jev_agent_to_handoff_runner():
    calls = []

    async def handoff(**kwargs):
        calls.append(kwargs)
        return True

    player = MacroPlayer(
        delay_between_actions=0,
        on_mismatch="agent",
        state_timeout=0.01,
        state_poll_interval=0.01,
        handoff_runner=handoff,
        agent="jev",
    )
    player.driver = FakeAndroidDriver(_android_state(text="Delete"))

    success = asyncio.run(
        player.replay_macro(_macro(normalize_ui_state(_android_state())))
    )

    assert success
    assert player.driver.taps == []
    assert len(calls) == 1
    assert calls[0]["agent"] == "jev"
    assert calls[0]["goal"] == "finish task"
    assert calls[0]["divergence"]["step"] == 1
    assert "ios" not in calls[0]


def test_handoff_runner_without_agent_or_ios_params_still_works():
    calls = []

    async def legacy_handoff(
        *,
        goal,
        device_serial,
        provider,
        model,
        config_path,
        divergence,
        current_state,
        remaining_actions,
    ):
        calls.append(goal)
        return True

    player = MacroPlayer(
        device_serial="emulator-5554",
        delay_between_actions=0,
        on_mismatch="agent",
        state_timeout=0.01,
        state_poll_interval=0.01,
        handoff_runner=legacy_handoff,
    )
    player.driver = FakeAndroidDriver(_android_state(text="Delete"))

    assert asyncio.run(
        player.replay_macro(_macro(normalize_ui_state(_android_state())))
    )
    assert calls == ["finish task"]


def test_run_agent_handoff_forwards_agent_and_ios_to_run_command(monkeypatch):
    import mobilerun.cli.main as cli_main

    calls = []

    async def fake_run_command(command, **kwargs):
        calls.append((command, kwargs))
        return True

    monkeypatch.setattr(cli_main, "run_command", fake_run_command)

    success = asyncio.run(
        run_agent_handoff(
            goal="finish task",
            device_serial=PORTAL_URL,
            config_path="jev.yaml",
            divergence={"step": 1},
            current_state={},
            remaining_actions=[],
            agent="jev",
            ios=True,
        )
    )

    assert success
    command, kwargs = calls[0]
    assert command == "finish task"
    assert kwargs["agent"] == "jev"
    assert kwargs["ios"] is True
    assert kwargs["device"] == PORTAL_URL
    assert kwargs["config_path"] == "jev.yaml"
    assert kwargs["save_trajectory"] == "none"


def test_run_agent_handoff_defaults_keep_configured_agent(monkeypatch):
    import mobilerun.cli.main as cli_main

    calls = []

    async def fake_run_command(command, **kwargs):
        calls.append(kwargs)
        return False

    monkeypatch.setattr(cli_main, "run_command", fake_run_command)

    assert not asyncio.run(
        run_agent_handoff(
            goal="g",
            device_serial="emulator-5554",
            divergence={},
            current_state={},
            remaining_actions=[],
        )
    )
    assert calls[0]["agent"] is None
    assert calls[0]["ios"] is False


@pytest.mark.parametrize(
    ("config_agent", "handoff_agent"), [("jev", None), ("mobilerun", "jev")]
)
def test_jev_handoff_constructs_mobile_agent_with_jev_config(
    monkeypatch, tmp_path, config_agent, handoff_agent
):
    import mobilerun.cli.main as cli_main

    monkeypatch.setattr(env_keys, "AUTH_PROFILES_PATH", tmp_path / "auth-profiles.json")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-typesafe-key")
    created = []

    class FakeHandler:
        async def stream_events(self):
            if False:
                yield None

        def __await__(self):
            async def done():
                return SimpleNamespace(success=True)

            return done().__await__()

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)

        def run(self):
            return FakeHandler()

    config = MobileConfig(
        agent=AgentConfig(name=config_agent),
        telemetry=TelemetryConfig(enabled=False),
    )
    monkeypatch.setattr(cli_main.ConfigLoader, "load", lambda *a, **k: config)
    monkeypatch.setattr(cli_main, "MobileAgent", FakeAgent)

    success = asyncio.run(
        run_agent_handoff(
            goal="finish task",
            device_serial=PORTAL_URL,
            divergence={"step": 1},
            current_state={},
            remaining_actions=[],
            agent=handoff_agent,
            ios=True,
        )
    )

    assert success
    assert created[0]["config"].agent.name == "jev"


def test_cli_replay_threads_agent_to_macro_player(monkeypatch, tmp_path):
    macro_file = tmp_path / "macro.json"
    macro_file.write_text(json.dumps(_macro(None)))
    players = []

    class FakePlayer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            players.append(self)

        def load_macro_from_file(self, path):
            return json.loads(macro_file.read_text())

        async def replay_macro(self, macro_data, **kwargs):
            return True

    monkeypatch.setattr(macro_cli_module, "MacroPlayer", FakePlayer)

    result = CliRunner().invoke(
        macro_cli_module.macro_cli,
        [
            "replay",
            str(macro_file),
            "-d",
            "emulator-5554",
            "--agent",
            "JEV",
            "--on-mismatch",
            "agent",
            "--config",
            "jev.yaml",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = players[0].kwargs
    assert kwargs["agent"] == "jev"
    assert kwargs["on_mismatch"] == "agent"
    assert kwargs["config_path"] == "jev.yaml"
    assert kwargs["device_serial"] == "emulator-5554"
    assert kwargs["ios"] is False


def test_cli_replay_rejects_unknown_agent(tmp_path):
    result = CliRunner().invoke(
        macro_cli_module.macro_cli,
        ["replay", str(tmp_path), "--agent", "other", "--dry-run"],
    )

    assert result.exit_code == 2
    assert "other" in result.output


def test_cli_ios_replay_discovers_portal_and_uses_android_tree_provider(
    monkeypatch, tmp_path, fake_ios_portal
):
    monkeypatch.setenv("MOBILERUN_DEVICE_TOKEN", "device-token")
    macro_file = tmp_path / "macro.json"
    macro_file.write_text(json.dumps(_macro(_portal_snapshot())))
    players = []

    class RecordingPlayer(MacroPlayer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            players.append(self)

    monkeypatch.setattr(macro_cli_module, "MacroPlayer", RecordingPlayer)

    result = CliRunner().invoke(
        macro_cli_module.macro_cli,
        ["replay", str(macro_file), "--ios", "-t", "0", "--state-timeout", "0.5"],
    )

    assert result.exit_code == 0, result.output
    assert fake_ios_portal["discovered"]
    assert fake_ios_portal["url"] == PORTAL_URL
    assert fake_ios_portal["token"] == "device-token"
    driver = fake_ios_portal["driver"]
    assert driver.connected
    assert driver.taps == [(60, 40)]
    player = players[0]
    assert player.ios is True
    assert player.device_serial == PORTAL_URL
    provider = player.state_provider
    assert isinstance(provider, AndroidStateProvider)
    assert isinstance(provider.tree_filter, DetailedFilter)
    assert isinstance(provider.tree_formatter, IndexedFormatter)
    assert provider.use_normalized is False


def test_cli_ios_replay_rejects_non_url_device(tmp_path, fake_ios_portal):
    macro_file = tmp_path / "macro.json"
    macro_file.write_text(json.dumps(_macro(None)))

    result = CliRunner().invoke(
        macro_cli_module.macro_cli,
        ["replay", str(macro_file), "--ios", "-d", "emulator-5554"],
    )

    assert result.exit_code == 1
    assert "portal base URL" in result.output
    assert "driver" not in fake_ios_portal


def test_ios_divergence_hands_off_with_ios_and_portal_url(fake_ios_portal):
    calls = []

    async def handoff(**kwargs):
        calls.append(kwargs)
        return True

    changed = dict(LOCAL_PORTAL_STATE)
    changed["a11y_tree"] = {
        **LOCAL_PORTAL_STATE["a11y_tree"],
        "children": [
            {
                **LOCAL_PORTAL_STATE["a11y_tree"]["children"][0],
                "text": "Delete Account",
                "resourceId": "Delete",
            }
        ],
    }
    player = MacroPlayer(
        device_serial=PORTAL_URL + "/",
        delay_between_actions=0,
        on_mismatch="agent",
        state_timeout=0.01,
        state_poll_interval=0.01,
        handoff_runner=handoff,
        agent="jev",
        ios=True,
    )
    player.driver = FakePortalHttpDriver(changed)

    assert asyncio.run(player.replay_macro(_macro(_portal_snapshot())))
    assert player.driver.taps == []
    assert calls[0]["ios"] is True
    assert calls[0]["agent"] == "jev"

    # Without an injected driver the resolved portal URL is handed off.
    calls.clear()
    player = MacroPlayer(
        device_serial=PORTAL_URL + "/",
        delay_between_actions=0,
        on_mismatch="agent",
        state_timeout=0.01,
        state_poll_interval=0.01,
        handoff_runner=handoff,
        ios=True,
    )
    asyncio.run(player.replay_macro(_macro(normalize_ui_state(_android_state()))))
    assert fake_ios_portal["url"] == PORTAL_URL
    assert calls[0]["device_serial"] == PORTAL_URL
    assert calls[0]["ios"] is True
    assert "agent" not in calls[0]


def test_legacy_ios_driver_snapshot_uses_ios_state_provider():
    player = MacroPlayer(driver=FakeLegacyIOSDriver())

    snapshot = asyncio.run(player.get_current_state_snapshot())

    assert isinstance(player.state_provider, IOSStateProvider)
    assert any(node["text"] == "General" for node in snapshot["nodes"])


def test_injected_state_provider_is_used():
    class FakeProvider:
        async def get_state(self):
            return SimpleNamespace(
                elements=[{"text": "Injected", "bounds": "0,0,10,10"}],
                phone_state={},
                screen_width=10,
                screen_height=10,
            )

    provider = FakeProvider()
    player = MacroPlayer(
        driver=FakeAndroidDriver(_android_state()), state_provider=provider
    )

    snapshot = asyncio.run(player.get_current_state_snapshot())

    assert player.state_provider is provider
    assert snapshot["nodes"][0]["text"] == "Injected"


def test_replay_macro_file_passes_handoff_options(monkeypatch, tmp_path):
    macro_file = tmp_path / "macro.json"
    macro_file.write_text(json.dumps(_macro(None)))
    players = []

    class RecordingPlayer(MacroPlayer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            players.append(self)

        async def replay_macro(self, macro_data, **kwargs):
            return True

    monkeypatch.setattr(replay_module, "MacroPlayer", RecordingPlayer)

    assert asyncio.run(
        replay_module.replay_macro_file(
            str(macro_file),
            device_serial=PORTAL_URL,
            on_mismatch="agent",
            config_path="jev.yaml",
            model="jev-1.2.3",
            agent="jev",
            ios=True,
        )
    )
    player = players[0]
    assert (player.agent, player.ios, player.config_path, player.model) == (
        "jev",
        True,
        "jev.yaml",
        "jev-1.2.3",
    )
