"""`mobilerun macro replay` must exit non-zero when the replay fails."""

import pytest
from click.testing import CliRunner

from mobilerun.macro import cli as macro_cli_module


class _FakePlayer:
    macro = {"macro_schema_version": "2.0", "description": "test", "actions": []}
    result = True
    error = None

    def __init__(self, **kwargs) -> None:
        pass

    def load_macro_from_folder(self, path):
        return self.macro

    async def replay_macro(self, macro_data, start_from_step=0, max_steps=None):
        if self.error:
            raise self.error
        return self.result


def _replay(monkeypatch, tmp_path, **player_attrs):
    player = type("Player", (_FakePlayer,), player_attrs)
    monkeypatch.setattr(macro_cli_module, "MacroPlayer", player)
    return CliRunner().invoke(
        macro_cli_module.macro_cli,
        ["replay", str(tmp_path), "--device", "emulator-5554"],
    )


def test_successful_replay_exits_zero(monkeypatch, tmp_path):
    result = _replay(monkeypatch, tmp_path)

    assert result.exit_code == 0


@pytest.mark.parametrize(
    "player_attrs",
    [
        pytest.param({"result": False}, id="replay-failed"),
        pytest.param({"macro": None}, id="macro-not-loaded"),
        pytest.param({"error": ValueError("unsupported schema")}, id="replay-raised"),
    ],
)
def test_failed_replay_exits_non_zero(monkeypatch, tmp_path, player_attrs):
    result = _replay(monkeypatch, tmp_path, **player_attrs)

    assert result.exit_code == 1
