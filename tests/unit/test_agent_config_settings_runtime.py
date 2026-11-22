from mobilerun.config import settings


def test_settings_defaults_and_runtime_precedence(monkeypatch):
    monkeypatch.setenv("MOBILERUN_CONFIG", "new.yaml")
    assert settings.get_settings().MOBILERUN_CONFIG == "new.yaml"


def test_empty_primary_config_is_preserved(monkeypatch):
    monkeypatch.setenv("MOBILERUN_CONFIG", "")
    assert settings.get_settings().MOBILERUN_CONFIG == ""


def test_legacy_config_environment_variable_is_ignored(monkeypatch):
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)
    monkeypatch.setenv("DROIDRUN_CONFIG", "legacy.yaml")
    assert settings.get_settings().MOBILERUN_CONFIG is None
