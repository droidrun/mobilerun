"""Environment snapshots preserve the shared runtime configuration rules."""

from mobilerun.config import settings


def test_factories_return_fresh_snapshots(monkeypatch):
    monkeypatch.setenv("MOBILERUN_CONFIG", "first.yaml")
    old_settings = settings.get_settings()
    monkeypatch.setenv("MOBILERUN_CONFIG", "second.yaml")
    assert old_settings.MOBILERUN_CONFIG == "first.yaml"
    assert settings.get_settings().MOBILERUN_CONFIG == "second.yaml"


def test_nonflag_empty_values_and_defaults(monkeypatch):
    for name in ("MOBILERUN_CONFIG", "MOBILERUN_AUTH_URL", "MOBILERUN_CLOUD_BASE_URL"):
        monkeypatch.setenv(name, "")
    snapshot = settings.get_settings()
    assert snapshot.MOBILERUN_CONFIG == ""
    assert snapshot.MOBILERUN_AUTH_URL == "https://cloud.mobilerun.ai/api/auth"
    assert snapshot.MOBILERUN_CLOUD_BASE_URL == "https://api.mobilerun.ai/v1"


def test_prefixed_tunables_and_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("MOBILERUN_CLOUD_API_TIMEOUT", "23.5")
    monkeypatch.setenv("MOBILERUN_DEVICE_AUTH_REQUEST_TIMEOUT", "17")
    monkeypatch.setenv("MOBILERUN_DEVICE_AUTH_MIN_POLL_INTERVAL", "2")
    monkeypatch.setenv("MOBILERUN_DEVICE_AUTH_SLOW_DOWN_BUMP", "3")
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("MOBILERUN_APP_NAME", "not-the-app-name")
    snapshot = settings.get_settings()
    assert snapshot.MOBILERUN_CLOUD_API_TIMEOUT == 23.5
    assert snapshot.MOBILERUN_DEVICE_AUTH_REQUEST_TIMEOUT == 17
    assert snapshot.MOBILERUN_DEVICE_AUTH_MIN_POLL_INTERVAL == 2
    assert snapshot.MOBILERUN_DEVICE_AUTH_SLOW_DOWN_BUMP == 3
    assert (
        snapshot.mobilerun_cloud_credential_path
        == tmp_path / "credentials" / "mobilerun-cloud.json"
    )
    assert "app_name" not in settings.Settings.model_fields


def test_explicit_aliases_bypass_prefix_and_remain_case_sensitive(monkeypatch):
    monkeypatch.setenv("MOBILERUN_MOBILERUN_CONFIG", "doubly-prefixed-wrong")
    monkeypatch.setenv("MOBILERUN_CONFIG", "exact.yaml")
    monkeypatch.setenv("MOBILERUN_CLOUD_API_TIMEOUT", "14.5")
    snapshot = settings.get_settings()
    assert snapshot.MOBILERUN_CONFIG == "exact.yaml"
    assert snapshot.MOBILERUN_CLOUD_API_TIMEOUT == 14.5
