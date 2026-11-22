"""Shared device-auth protocol values and credential locations."""

from pathlib import Path

from mobilerun.config import settings


def test_device_auth_protocol_defaults():
    snapshot = settings.get_settings()
    assert snapshot.MOBILERUN_DEVICE_AUTH_REQUEST_TIMEOUT == 10.0
    assert snapshot.MOBILERUN_DEVICE_AUTH_MIN_POLL_INTERVAL == 5.0
    assert snapshot.MOBILERUN_DEVICE_AUTH_SLOW_DOWN_BUMP == 5.0


def test_cloud_credential_path_reads_settings_at_call_time(monkeypatch, tmp_path):
    target = tmp_path / "credentials" / "mobilerun-cloud.json"
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))
    assert settings.get_settings().mobilerun_cloud_credential_path == target
    assert isinstance(settings.get_settings().mobilerun_cloud_credential_path, Path)


def test_configurable_defaults_are_typed_fields_and_read_fresh(monkeypatch):
    assert settings.get_settings().MOBILERUN_PORTAL_REMOTE_PORT == 8080
    assert (
        settings.get_settings().MOBILERUN_AUTH_URL
        == "https://cloud.mobilerun.ai/api/auth"
    )
    assert (
        settings.get_settings().MOBILERUN_CLOUD_BASE_URL
        == "https://api.mobilerun.ai/v1"
    )
    monkeypatch.setenv("MOBILERUN_PORTAL_REMOTE_PORT", "9090")
    monkeypatch.setenv("MOBILERUN_AUTH_URL", "https://auth.example")
    monkeypatch.setenv("MOBILERUN_CLOUD_BASE_URL", "https://cloud.example")
    monkeypatch.setenv(
        "MOBILERUN_AGENTS", '{"custom":{"skills":["global","project"],"mcp":null}}'
    )
    current = settings.get_settings()
    assert current.MOBILERUN_PORTAL_REMOTE_PORT == 9090
    assert current.MOBILERUN_AUTH_URL == "https://auth.example"
    assert current.MOBILERUN_CLOUD_BASE_URL == "https://cloud.example"
    assert current.MOBILERUN_AGENTS["custom"].skills == ("global", "project")
