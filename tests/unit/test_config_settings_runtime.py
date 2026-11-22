from mobilerun.config import settings


def test_core_settings_are_case_sensitive_and_ignore_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)
    monkeypatch.setenv("mobilerun_config", "lowercase.yaml")
    monkeypatch.setenv("UNRELATED_SETTING", "not valid")
    (tmp_path / ".env").write_text("MOBILERUN_CONFIG=dotenv.yaml\n")
    assert settings.Settings().MOBILERUN_CONFIG is None
    assert settings.get_settings().MOBILERUN_CONFIG is None
