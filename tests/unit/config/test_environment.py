"""Environment settings are read on demand; secret snapshots mask their values."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mobilerun.config import env, settings


@pytest.mark.parametrize(
    ("name", "default", "override", "invalid"),
    [
        ("MOBILERUN_WEBHOOK_PAGE_SIZE", 100, 1, ("0", "101")),
        ("MOBILERUN_WEBHOOK_PING_TIMEOUT", 15.0, 60.0, ("0", "60.1")),
    ],
)
def test_webhook_settings(monkeypatch, name, default, override, invalid):
    monkeypatch.delenv(name, raising=False)
    assert getattr(settings.get_settings(), name) == default
    monkeypatch.setenv(name, str(override))
    configured = getattr(settings.get_settings(), name)
    assert type(configured) is type(override)
    assert configured == override
    for value in invalid:
        monkeypatch.setenv(name, value)
        with pytest.raises(ValidationError) as error:
            settings.get_settings()
        assert [detail["loc"] for detail in error.value.errors()] == [(name,)]


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


def test_explicit_aliases_bypass_prefix_and_remain_case_sensitive(monkeypatch):
    monkeypatch.setenv("MOBILERUN_MOBILERUN_CONFIG", "doubly-prefixed-wrong")
    monkeypatch.setenv("MOBILERUN_CONFIG", "exact.yaml")
    monkeypatch.setenv("MOBILERUN_CLOUD_API_TIMEOUT", "14.5")
    snapshot = settings.get_settings()
    assert snapshot.MOBILERUN_CONFIG == "exact.yaml"
    assert snapshot.MOBILERUN_CLOUD_API_TIMEOUT == 14.5


def test_legacy_config_environment_variable_is_ignored(monkeypatch):
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)
    monkeypatch.setenv("DROIDRUN_CONFIG", "legacy.yaml")
    assert settings.get_settings().MOBILERUN_CONFIG is None


def test_core_settings_are_case_sensitive_and_ignore_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)
    monkeypatch.setenv("mobilerun_config", "lowercase.yaml")
    monkeypatch.setenv("UNRELATED_SETTING", "not valid")
    (tmp_path / ".env").write_text("MOBILERUN_CONFIG=dotenv.yaml\n")
    assert settings.Settings().MOBILERUN_CONFIG is None
    assert settings.get_settings().MOBILERUN_CONFIG is None


def test_device_auth_protocol_defaults():
    snapshot = settings.get_settings()
    assert snapshot.MOBILERUN_DEVICE_AUTH_REQUEST_TIMEOUT == 10.0
    assert snapshot.MOBILERUN_DEVICE_AUTH_MIN_POLL_INTERVAL == 5.0
    assert snapshot.MOBILERUN_DEVICE_AUTH_SLOW_DOWN_BUMP == 5.0


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


def test_cloud_credential_path_reads_settings_at_call_time(monkeypatch, tmp_path):
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))
    path = settings.get_settings().mobilerun_cloud_credential_path
    assert path == tmp_path / "credentials" / "mobilerun-cloud.json"
    assert isinstance(path, Path)


def test_sdk_paths_read_at_call_time(monkeypatch):
    monkeypatch.setenv("ANDROID_HOME", "/custom/sdk")
    assert str(settings.get_settings().ANDROID_HOME) == "/custom/sdk"


def test_configurable_defaults_are_typed_fields_and_read_fresh(monkeypatch):
    defaults = settings.get_settings()
    assert defaults.MOBILERUN_PORTAL_REMOTE_PORT == 8080
    assert defaults.MOBILERUN_AUTH_URL == "https://cloud.mobilerun.ai/api/auth"
    assert defaults.MOBILERUN_CLOUD_BASE_URL == "https://api.mobilerun.ai/v1"
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


def test_base_secrets_cover_cloud_environment(monkeypatch):
    monkeypatch.setenv("MOBILERUN_API_KEY", "secret-placeholder-one")
    monkeypatch.setenv("MOBILERUN_DEVICE_TOKEN", "two")
    snapshot = env.get_secrets()
    assert snapshot.MOBILERUN_API_KEY.get_secret_value() == "secret-placeholder-one"
    assert env.secret_value(snapshot.MOBILERUN_API_KEY) == "secret-placeholder-one"
    assert snapshot.MOBILERUN_DEVICE_TOKEN.get_secret_value() == "two"
    assert "secret-placeholder-one" not in repr(snapshot)
    monkeypatch.setenv("MOBILERUN_API_KEY", "secret-placeholder-changed")
    assert (
        env.get_secrets().MOBILERUN_API_KEY.get_secret_value()
        == "secret-placeholder-changed"
    )


@pytest.mark.parametrize("name", env.Secrets.model_fields)
def test_secrets_are_aliased_and_masked(monkeypatch, name):
    monkeypatch.setenv(name, "sensitive-placeholder")
    snapshot = env.get_secrets()
    field = getattr(snapshot, name)
    assert field.get_secret_value() == "sensitive-placeholder"
    assert "sensitive-placeholder" not in str(field)
    assert "sensitive-placeholder" not in repr(field)
    assert "sensitive-placeholder" not in repr(snapshot)
    assert "sensitive-placeholder" not in str(snapshot)
