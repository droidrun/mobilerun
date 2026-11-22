"""Shared config persistence stays independent of typed optional sections."""

from copy import deepcopy

from mobilerun.config.user_config import MobileConfig


def test_shared_config_round_trips_raw_optional_sections_without_mutating_input():
    data = {
        "_version": 8,
        "device": {"serial": "test"},
        "agent": {"max_steps": 3, "manager": {"stateless": True}},
        "llm_profiles": {"manager": {"provider": "GoogleGenAI", "model": "test"}},
        "mcp": {"enabled": True, "servers": {"local": {"command": "echo"}}},
        "credentials": {"enabled": True, "file_path": "credentials.yaml"},
        "tools": {"disabled_tools": []},
    }
    original = deepcopy(data)
    shared = MobileConfig.from_dict(data)
    for section in ("agent", "llm_profiles", "mcp", "credentials", "tools"):
        assert shared.to_dict()[section] == data[section]
    assert data == original
