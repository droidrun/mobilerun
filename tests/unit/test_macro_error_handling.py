import asyncio

import pytest

from mobilerun.macro.errors import MacroError, MacroMismatch
from mobilerun.macro.player import MacroPlayer, replay_macro_file


def test_unknown_action_raises_macro_error():
    player = MacroPlayer(device=object())
    with pytest.raises(MacroError):
        asyncio.run(player.replay_action({"type": "unknown"}))


def test_start_app_without_package_raises_clear_macro_error():
    player = MacroPlayer(device=object())
    with pytest.raises(MacroError, match="start_app.*package"):
        asyncio.run(player.replay_action({"type": "start_app"}))


def test_state_programmer_error_is_not_swallowed():
    class Provider:
        async def get_state(self):
            raise TypeError("programmer bug")

    player = MacroPlayer(device=object())
    player.state_provider = Provider()
    with pytest.raises(TypeError, match="programmer bug"):
        asyncio.run(player.get_current_state_snapshot())


def test_file_helper_propagates_mismatch(monkeypatch):
    mismatch = MacroMismatch(0, [], {}, "mismatch")
    monkeypatch.setattr(MacroPlayer, "load_macro_from_file", lambda *args: {})

    async def replay(*args, **kwargs):
        raise mismatch

    monkeypatch.setattr(MacroPlayer, "replay_macro", replay)
    with pytest.raises(MacroMismatch) as caught:
        asyncio.run(replay_macro_file("unused"))
    assert caught.value is mismatch
