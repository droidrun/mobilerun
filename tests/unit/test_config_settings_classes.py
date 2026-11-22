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


def test_registry_and_oauth_defaults_follow_environment_after_import(
    monkeypatch, tmp_path
):
    pytest.importorskip("mobilerun.agents", reason="agent providers belong to add-on")
    from mobilerun.agents.agent.providers.registry import (
        list_provider_families,
        resolve_provider_variant,
    )
    from mobilerun.agents.agent.utils.oauth.anthropic_oauth_llm import AnthropicOAuthLLM
    from mobilerun.agents.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        GeminiOAuthCodeAssistLLM,
    )
    from mobilerun.agents.agent.utils.oauth.grok_oauth_llm import (
        GrokOAuthCredentialStore,
    )
    from mobilerun.agents.agent.utils.oauth.openai_oauth_llm import (
        OpenAIOAuthCredentialStore,
    )

    explicit_path = tmp_path / "explicit.json"
    for directory in (tmp_path / "first", tmp_path / "second"):
        monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(directory))
        expected = directory / "credentials" / "auth-profiles.json"
        for family in ("gemini", "openai", "anthropic", "xai"):
            assert resolve_provider_variant(family, "oauth").credential_path == str(
                expected
            )
        for family in list_provider_families():
            for variant in family.variants:
                if variant.auth_mode == "oauth":
                    assert variant.credential_path == str(expected)
        for store_class in (OpenAIOAuthCredentialStore, GrokOAuthCredentialStore):
            assert store_class().path == expected
            assert store_class(explicit_path).path == explicit_path
        for llm_class in (AnthropicOAuthLLM, GeminiOAuthCodeAssistLLM):
            assert llm_class.model_fields["credential_path"].get_default(
                call_default_factory=True
            ) == str(expected)
            assert llm_class(access_token="placeholder").credential_path == str(
                expected
            )
            assert llm_class(
                access_token="placeholder", credential_path=str(explicit_path)
            ).credential_path == str(explicit_path)


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
