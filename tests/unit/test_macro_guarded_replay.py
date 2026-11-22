import asyncio
from types import SimpleNamespace

import pytest

from mobilerun.core.credentials import DictCredentials
from mobilerun.core.devices.base import Device as DeviceMixin
from mobilerun.macro import MacroMismatch
from mobilerun.macro.models import normalize_ui_state
from mobilerun.macro.player import MacroPlayer


class FakeDriver(DeviceMixin):
    platform = "android"
    state = None

    def __init__(self, states):
        super().__init__()
        self.states = list(states)
        self.taps = []
        self.inputs = []

    async def connect(self):
        return None

    async def _get_ui_tree(self):
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]

    async def tap(self, x, y):
        self.taps.append((x, y))

    async def input_text(self, text, clear=False):
        self.inputs.append((text, clear))
        return True

    async def _input_text_raw(self, text, clear=False, stealth=False, wpm=0):
        return await self.input_text(text, clear)


def _raw_state(text="Continue", bounds="10,20,110,60"):
    return {
        "a11y_tree": [
            {
                "resourceId": "com.example:id/continue",
                "className": "android.widget.Button",
                "text": text,
                "clickable": True,
                "enabled": True,
                "bounds": bounds,
            }
        ],
        "phone_state": {"package": "com.example", "activity": ".MainActivity"},
        "screen_width": 400,
        "screen_height": 800,
    }


def _macro_action(pre_state, action_type="tap"):
    action = {
        "action_type": action_type,
        "pre_state": pre_state,
        "recorded_at_ms": 1000,
        "elapsed_since_previous_ms": 0,
    }
    if action_type == "tap":
        action.update(
            {
                "x": 60,
                "y": 40,
            }
        )
    return action


def test_guarded_replay_waits_until_expected_state_appears_then_taps_saved_coordinates():
    saved_state = normalize_ui_state(_raw_state())
    player = MacroPlayer(
        delay_between_actions=0,
        state_timeout=0.5,
        state_poll_interval=0.01,
    )
    player.driver = FakeDriver(
        [_raw_state(text="Loading"), _raw_state(bounds="50,100,250,180")]
    )
    macro_data = {
        "macro_schema_version": "2.0",
        "description": "continue",
        "actions": [_macro_action(saved_state)],
    }

    success = asyncio.run(player.replay_macro(macro_data))

    assert success
    assert player.driver.taps == [(60, 40)]


def test_replay_ignores_stale_target_hint_and_uses_recorded_coordinates():
    saved_state = normalize_ui_state(_raw_state())
    action = _macro_action(saved_state)
    action["target_hint"] = saved_state["nodes"][0]
    player = MacroPlayer(
        delay_between_actions=0,
        state_timeout=0.5,
        state_poll_interval=0.01,
    )
    player.driver = FakeDriver([_raw_state(bounds="50,100,250,180")])
    macro_data = {
        "macro_schema_version": "2.0",
        "description": "continue",
        "actions": [action],
    }

    success = asyncio.run(player.replay_macro(macro_data))

    assert success
    assert player.driver.taps == [(60, 40)]


def test_stop_mode_aborts_on_state_mismatch_with_divergence_report():
    saved_state = normalize_ui_state(_raw_state())
    player = MacroPlayer(
        delay_between_actions=0,
        state_timeout=0.01,
        state_poll_interval=0.01,
    )
    player.driver = FakeDriver([_raw_state(text="Delete")])
    macro_data = {
        "macro_schema_version": "2.0",
        "description": "continue",
        "actions": [_macro_action(saved_state)],
    }

    with pytest.raises(MacroMismatch):
        asyncio.run(player.replay_macro(macro_data))
    assert player.driver.taps == []
    assert player.last_divergence["step"] == 1
    assert "below threshold" in player.last_divergence["reason"]


def test_current_state_snapshot_requires_device_ui_state():
    class IncompleteDevice:
        async def _get_ui_tree(self):
            return {}

    player = MacroPlayer(IncompleteDevice())
    with pytest.raises(AttributeError, match="state"):
        asyncio.run(player.get_current_state_snapshot())


def test_current_state_snapshot_prefers_state_provider_when_available():
    class FakeProvider:
        async def get_state(self):
            return SimpleNamespace(
                elements=[
                    {
                        "resourceId": "com.example:id/provider",
                        "className": "android.widget.TextView",
                        "text": "Provider formatted state",
                        "bounds": "0,0,100,100",
                    }
                ],
                phone_state={"package": "com.example", "activity": ".Main"},
                screen_width=100,
                screen_height=100,
            )

    player = MacroPlayer(delay_between_actions=0)
    player.driver = FakeDriver([_raw_state(text="Raw state")])
    player.state_provider = FakeProvider()

    snapshot = asyncio.run(player.get_current_state_snapshot())

    assert snapshot["nodes"][0]["text"] == "Provider formatted state"


def test_type_secret_replay_resolves_secret_without_logging_value():
    from unittest.mock import AsyncMock

    saved_state = normalize_ui_state(_raw_state())
    player = MacroPlayer(delay_between_actions=0)
    player.driver = FakeDriver([_raw_state()])
    player.driver.state = SimpleNamespace(
        refresh=AsyncMock(
            return_value=SimpleNamespace(
                phone_state={
                    "focusedElement": {"className": "EditText"},
                    "isEditable": True,
                }
            )
        )
    )
    player.state_provider = SimpleNamespace(
        get_state=AsyncMock(return_value=_raw_state())
    )
    asyncio.run(DictCredentials({"login_password": "s3cr3t"}).bind(player.driver))
    macro_data = {
        "macro_schema_version": "2.0",
        "description": "type secret",
        "actions": [
            {
                "action_type": "type_secret",
                "secret_id": "login_password",
                "clear": False,
                "pre_state": saved_state,
            }
        ],
    }

    success = asyncio.run(player.replay_macro(macro_data))

    assert success
    assert player.driver.inputs == [("s3cr3t", False)]


def test_type_secret_replay_fails_without_typing_when_secret_missing():
    saved_state = normalize_ui_state(_raw_state())
    player = MacroPlayer(delay_between_actions=0)
    player.driver = FakeDriver([_raw_state()])
    assert not player.driver.extensions
    macro_data = {
        "macro_schema_version": "2.0",
        "description": "type secret",
        "actions": [
            {
                "action_type": "type_secret",
                "secret_id": "missing_password",
                "clear": False,
                "pre_state": saved_state,
            }
        ],
    }

    from mobilerun.errors import MacroError

    with pytest.raises(MacroError, match="Bind one vault"):
        asyncio.run(player.replay_macro(macro_data))
    assert player.driver.inputs == []


def test_type_secret_replay_does_not_load_environment_config(tmp_path, monkeypatch):
    import yaml

    secrets = tmp_path / "secrets.yaml"
    secrets.write_text(yaml.safe_dump({"secrets": {"login": "test-value"}}))
    config = tmp_path / "custom.yaml"
    config.write_text(
        yaml.safe_dump(
            {"_version": 8, "credentials": {"enabled": True, "file_path": str(secrets)}}
        )
    )
    monkeypatch.setenv("MOBILERUN_CONFIG", str(config))
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path / "missing"))
    device = FakeDriver([_raw_state()])
    player = MacroPlayer(device)
    from mobilerun.errors import MacroError

    with pytest.raises(MacroError, match="Bind one vault"):
        asyncio.run(
            player.replay_action({"action_type": "type_secret", "secret_id": "login"})
        )
    assert device.inputs == []
