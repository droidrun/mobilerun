"""Previously removed module paths remain unavailable."""

import importlib.util

import pytest


@pytest.mark.parametrize("suffix", ["telemetry", "credential_manager"])
def test_unadvertised_module_path_is_absent(suffix):
    assert importlib.util.find_spec("mobilerun." + suffix) is None


def test_legacy_mcp_modules_are_absent():
    assert importlib.util.find_spec("mobilerun.mcp.adapter") is None
    assert importlib.util.find_spec("mobilerun.mcp.client") is None
