"""Offline diagnostics and repair fixture for doctor CLI tests."""

import importlib
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from mobilerun.diagnostics.results import CheckResult, Status

__all__ = ["checks"]


@pytest.fixture
def checks(monkeypatch):
    module = importlib.import_module("mobilerun.cli.doctor")
    ok = CheckResult("test", Status.PASS, "ready")
    fail = CheckResult("Portal", Status.FAIL, "not installed")
    warn = CheckResult("TCP", Status.WARN, "not configured")
    for name in (
        "check_sdk_version",
        "check_adb",
        "check_content_provider",
        "check_keyboard",
    ):
        monkeypatch.setattr(module, name, AsyncMock(return_value=ok))
    monkeypatch.setattr(module, "check_config", AsyncMock(return_value=(ok, None)))
    monkeypatch.setattr(
        module, "check_device", AsyncMock(return_value=(ok, NS(serial="serial")))
    )
    monkeypatch.setattr(
        module, "check_portal_installed", AsyncMock(return_value=(fail, False))
    )
    monkeypatch.setattr(
        module, "check_portal_version", AsyncMock(return_value=(ok, "1", "1", None))
    )
    monkeypatch.setattr(
        module, "check_accessibility", AsyncMock(return_value=(ok, True))
    )
    monkeypatch.setattr(
        module, "check_portal_state", AsyncMock(return_value=(ok, None))
    )
    monkeypatch.setattr(module, "check_tcp", AsyncMock(return_value=warn))
    monkeypatch.setattr(module, "install_portal", AsyncMock())
    monkeypatch.setattr(
        module, "fix_enable_accessibility", AsyncMock(return_value=True)
    )
    return module, ok, fail
