import json
from io import StringIO
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from rich.console import Console

import mobilerun.cli.configure_wizard as configure_wizard
import mobilerun.cli.main as cli_main
from mobilerun.agent.providers.registry import resolve_provider_variant
from mobilerun.agent.utils.oauth.login_timeout import OAuthLoginDeadline
from mobilerun.cli import oauth_actions
from mobilerun.cli.configure_wizard import ConfigureWizardCallbacks
from mobilerun.config_manager import MobileConfig


def test_xai_oauth_credentials_are_detected_by_nested_slot(tmp_path) -> None:
    credential_path = tmp_path / "auth-profiles.json"
    credential_path.write_text(
        json.dumps(
            {
                "openaiOauth": {"access": "unrelated"},
                "grokOauth": {"access_token": "grok-access"},
            }
        ),
        encoding="utf-8",
    )

    assert configure_wizard._oauth_credentials_present(
        str(credential_path), "xai_oauth"
    )

    credential_path.write_text(
        json.dumps({"openaiOauth": {"access": "unrelated"}}),
        encoding="utf-8",
    )
    assert not configure_wizard._oauth_credentials_present(
        str(credential_path), "xai_oauth"
    )


def test_wizard_prepares_xai_oauth_with_selected_model(tmp_path) -> None:
    calls: list[dict] = []
    callbacks = ConfigureWizardCallbacks(
        run_openai_oauth_login=lambda **kwargs: None,
        run_anthropic_oauth_login=lambda **kwargs: None,
        run_gemini_oauth_login=lambda **kwargs: None,
        run_grok_oauth_login=lambda **kwargs: calls.append(kwargs),
    )

    configure_wizard._prepare_variant_auth(
        callbacks=callbacks,
        variant=SimpleNamespace(id="xai_oauth"),
        credential_path=str(tmp_path / "auth-profiles.json"),
        selected_model="grok-4.6",
    )

    assert calls == [
        {
            "credential_path": str(tmp_path / "auth-profiles.json"),
            "model": "grok-4.6",
        }
    ]


def test_configure_xai_command_forwards_device_code_options(
    monkeypatch, tmp_path
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        cli_main,
        "_run_grok_oauth_login",
        lambda **kwargs: calls.append(kwargs),
    )
    credential_path = tmp_path / "auth-profiles.json"

    result = CliRunner().invoke(
        cli_main.cli,
        [
            "configure",
            "xai",
            "--credential-path",
            str(credential_path),
            "--model",
            "grok-4.6",
            "--timeout",
            "12",
            "--no-browser",
            "--device-code",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "credential_path": str(credential_path),
            "model": "grok-4.6",
            "timeout": 12.0,
            "open_browser": False,
            "device_code": True,
        }
    ]


def test_configure_anthropic_forwards_timeout_and_browser_preference(
    monkeypatch, tmp_path
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        cli_main,
        "_run_anthropic_oauth_login",
        lambda **kwargs: calls.append(kwargs),
    )
    credential_path = tmp_path / "auth-profiles.json"

    result = CliRunner().invoke(
        cli_main.cli,
        [
            "configure",
            "anthropic",
            "--credential-path",
            str(credential_path),
            "--timeout",
            "12",
            "--no-browser",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "credential_path": str(credential_path),
            "timeout": 12.0,
            "open_browser": False,
        }
    ]


def test_configure_oauth_timeout_must_be_positive_and_finite() -> None:
    result = CliRunner().invoke(
        cli_main.cli,
        ["configure", "xai", "--timeout", "nan"],
    )

    assert result.exit_code == 2
    assert "finite number greater than zero" in result.output


def test_xai_oauth_action_shares_deadline_and_browser_preference(
    monkeypatch, tmp_path
) -> None:
    observed: dict = {}

    class FakeXAI:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            observed["init"] = kwargs

        def login(self, **kwargs):  # type: ignore[no-untyped-def]
            observed["login"] = kwargs

    monkeypatch.setattr(oauth_actions, "GrokOAuth", FakeXAI)
    oauth_actions.run_grok_oauth_login(
        str(tmp_path / "auth.json"),
        None,
        timeout=12,
        open_browser=False,
        device_code=True,
    )

    deadline = observed["login"]["deadline"]
    assert isinstance(deadline, OAuthLoginDeadline)
    assert observed["init"]["model"] == "grok-4.7"
    assert observed["init"]["timeout"] == 12
    assert observed["login"]["open_browser"] is False
    assert observed["login"]["device_code"] is True


def test_gemini_empty_entitlement_is_not_persisted(monkeypatch, tmp_path) -> None:
    persisted: list[OAuthLoginDeadline] = []

    class FakeGemini:
        def __init__(self, **kwargs):
            pass

        def login(self, **kwargs):
            return "access"

        def fetch_available_models(self, **kwargs):
            return []

        def _persist_credentials(self, *, deadline):
            persisted.append(deadline)

    monkeypatch.setattr(oauth_actions, "GeminiOAuthCodeAssistLLM", FakeGemini)
    with pytest.raises(RuntimeError, match="no usable models"):
        oauth_actions.run_gemini_oauth_login(
            str(tmp_path / "auth.json"),
            "gemini-3.7-flash-tiered",
            open_browser=False,
        )

    assert persisted == []


def test_configure_help_advertises_only_xai_provider_and_login_command() -> None:
    runner = CliRunner()

    run_help = runner.invoke(cli_main.cli, ["run", "--help"])
    configure_help = runner.invoke(cli_main.cli, ["configure", "--help"])
    xai_help = runner.invoke(cli_main.cli, ["configure", "xai", "--help"])
    removed_grok_command = runner.invoke(cli_main.cli, ["configure", "grok"])

    assert run_help.exit_code == 0
    assert "XAI" in run_help.output
    assert "xai_oauth" not in run_help.output
    assert "grok_oauth" not in run_help.output
    assert configure_help.exit_code == 0
    assert "xai" in configure_help.output.lower()
    assert "grok" not in configure_help.output.lower()
    assert xai_help.exit_code == 0
    assert "--device-code" in xai_help.output
    assert "xai" in xai_help.output.lower()
    assert "grok" not in xai_help.output.lower()
    assert removed_grok_command.exit_code != 0


@pytest.mark.parametrize(
    ("auth_mode", "expected_provider"),
    (("api_key", "XAI"), ("oauth", "xai_oauth")),
)
def test_exact_xai_configure_forms_keep_provider_and_auth_fixed(
    monkeypatch, auth_mode: str, expected_provider: str
) -> None:
    config = MobileConfig()
    saved_configs: list[MobileConfig] = []
    login_calls: list[dict] = []
    model_prompts: list[tuple[tuple[str, ...], str]] = []

    monkeypatch.setattr(configure_wizard.ConfigLoader, "load", lambda: config)
    monkeypatch.setattr(
        configure_wizard.ConfigLoader,
        "save",
        lambda saved: saved_configs.append(saved),
    )

    def choose_model(  # type: ignore[no-untyped-def]
        models, *, family_id, auth_mode, default_model, allow_back=True
    ):
        model_prompts.append((tuple(models), default_model))
        return default_model

    monkeypatch.setattr(configure_wizard, "_prompt_model_choice", choose_model)
    monkeypatch.setattr(
        configure_wizard,
        "_prompt_api_key_for_variant",
        lambda variant: ("xai-env-key", "env"),
    )
    monkeypatch.setattr(
        configure_wizard,
        "_oauth_credentials_present",
        lambda credential_path, variant_id: True,
    )
    monkeypatch.setattr(
        configure_wizard,
        "_prompt_oauth_credential_action",
        lambda credential_path: "use_existing",
    )
    monkeypatch.setattr(
        configure_wizard,
        "select_prompt",
        lambda *args, **kwargs: pytest.fail(
            "fixed XAI configure flow unexpectedly reopened the top-level menu"
        ),
    )
    monkeypatch.setattr(
        cli_main,
        "_run_grok_oauth_login",
        lambda **kwargs: login_calls.append(kwargs),
    )

    result = CliRunner().invoke(
        cli_main.cli,
        ["configure", "--provider", "XAI", "--auth-mode", auth_mode],
    )

    assert result.exit_code == 0, result.output
    assert saved_configs == [config]
    assert login_calls == []
    variant = resolve_provider_variant("xai", auth_mode)
    assert model_prompts == [(variant.models, variant.default_model)]
    assert {
        (profile.provider, profile.provider_family, profile.auth_mode, profile.model)
        for profile in config.llm_profiles.values()
    } == {(expected_provider, "xai", auth_mode, variant.default_model)}
    assert "xai_oauth" not in result.output
    assert "Provider: XAI" in result.output


def test_provider_only_flag_does_not_auto_enter_model_flow(monkeypatch) -> None:
    config = MobileConfig()
    saved_configs: list[MobileConfig] = []
    menu_calls: list[str] = []

    monkeypatch.setattr(configure_wizard.ConfigLoader, "load", lambda: config)
    monkeypatch.setattr(
        configure_wizard.ConfigLoader,
        "save",
        lambda saved: saved_configs.append(saved),
    )
    monkeypatch.setattr(
        configure_wizard,
        "_prompt_model_choice",
        lambda *args, **kwargs: pytest.fail(
            "a provider-only invocation must start at the top-level menu"
        ),
    )

    def choose_top_level(message, *args, **kwargs):  # type: ignore[no-untyped-def]
        menu_calls.append(message)
        return "finish"

    monkeypatch.setattr(configure_wizard, "select_prompt", choose_top_level)

    configure_wizard.run_configure_wizard(
        Console(file=StringIO(), force_terminal=False),
        ConfigureWizardCallbacks(
            run_openai_oauth_login=lambda **kwargs: None,
            run_anthropic_oauth_login=lambda **kwargs: None,
            run_gemini_oauth_login=lambda **kwargs: None,
        ),
        provider="ollama",
        auth_mode=None,
        model=None,
        api_key=None,
        base_url=None,
    )

    assert menu_calls == ["Configure"]
    assert saved_configs == [config]


def test_fixed_provider_and_auth_model_back_returns_to_top_level_once(
    monkeypatch,
) -> None:
    config = MobileConfig()
    saved_configs: list[MobileConfig] = []
    model_prompts: list[tuple[tuple[str, ...], str]] = []
    menu_calls: list[str] = []

    monkeypatch.setattr(configure_wizard.ConfigLoader, "load", lambda: config)
    monkeypatch.setattr(
        configure_wizard.ConfigLoader,
        "save",
        lambda saved: saved_configs.append(saved),
    )

    def choose_model(  # type: ignore[no-untyped-def]
        models, *, family_id, auth_mode, default_model, allow_back=True
    ):
        model_prompts.append((tuple(models), default_model))
        return configure_wizard._BACK

    def choose_top_level(message, *args, **kwargs):  # type: ignore[no-untyped-def]
        menu_calls.append(message)
        return "finish"

    monkeypatch.setattr(configure_wizard, "_prompt_model_choice", choose_model)
    monkeypatch.setattr(configure_wizard, "select_prompt", choose_top_level)

    configure_wizard.run_configure_wizard(
        Console(file=StringIO(), force_terminal=False),
        ConfigureWizardCallbacks(
            run_openai_oauth_login=lambda **kwargs: None,
            run_anthropic_oauth_login=lambda **kwargs: None,
            run_gemini_oauth_login=lambda **kwargs: None,
        ),
        provider="XAI",
        auth_mode="api_key",
        model=None,
        api_key=None,
        base_url=None,
    )

    assert model_prompts == [(("grok-4.6",), "grok-4.6")]
    assert menu_calls == ["Configure"]
    assert saved_configs == [config]
