"""The old module paths are intentionally unsupported."""

import importlib.util

import pytest


@pytest.mark.parametrize(
    "suffix",
    [
        "agent",
        "telemetry",
        "config_manager",
        "credential_manager",
        "tools",
    ],
)
def test_old_module_path_is_absent(suffix):
    assert importlib.util.find_spec("mobilerun." + suffix) is None


def test_legacy_mcp_modules_are_absent():
    assert importlib.util.find_spec("mobilerun.mcp.adapter") is None
    assert importlib.util.find_spec("mobilerun.mcp.client") is None
