"""Environment snapshots preserve the existing runtime configuration rules."""

import pytest

from mobilerun.config import env


@pytest.mark.parametrize("name", env.Secrets.model_fields)
def test_secrets_are_aliased_and_masked(monkeypatch, name):
    alias = name
    monkeypatch.setenv(alias, "sensitive-placeholder")
    snapshot = env.get_secrets()
    assert getattr(snapshot, name).get_secret_value() == "sensitive-placeholder"
    assert "sensitive-placeholder" not in str(getattr(snapshot, name))
    assert "sensitive-placeholder" not in repr(getattr(snapshot, name))
    assert "sensitive-placeholder" not in repr(snapshot)
    assert "sensitive-placeholder" not in str(snapshot)


def test_phoenix_names_ignore_lowercase(monkeypatch):
    pytest.importorskip("mobilerun.agents", reason="agent settings belong to add-on")
    monkeypatch.setenv("phoenix_project_name", "lowercase-project")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "uppercase-project")
    from mobilerun.agents.config.settings import get_agent_settings

    assert get_agent_settings().PHOENIX_PROJECT_NAME == "uppercase-project"


def test_saved_provider_keys_follow_environment_after_import(monkeypatch, tmp_path):
    pytest.importorskip("mobilerun.agents", reason="agent providers belong to add-on")
    import json

    from mobilerun.agents.clients.oauth.profiles import load_env_key_sources

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    for name in ("first", "second"):
        directory = tmp_path / name
        credentials = directory / "credentials"
        credentials.mkdir(parents=True)
        (credentials / "auth-profiles.json").write_text(
            json.dumps({"apiKeys": {"xai": name}})
        )
        monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(directory))
        assert load_env_key_sources()["xai"].saved == name
