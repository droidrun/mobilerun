from io import StringIO

import pytest
from rich.console import Console

import mobilerun.cli.configure_prompts as configure_prompts
import mobilerun.cli.configure_wizard as configure_wizard
from mobilerun.cli.configure_prompts import SelectChoice
from mobilerun.cli.configure_wizard import ConfigureWizardCallbacks
from mobilerun.config_manager import MobileConfig


def test_inquirer_select_displays_label_and_returns_canonical_value(
    monkeypatch,
) -> None:
    captured = {}

    class FakePrompt:
        def execute(self):
            return "gemini-3.8-flash-tiered"

    class FakeInquirer:
        def select(self, **kwargs):
            captured.update(kwargs)
            return FakePrompt()

    monkeypatch.setattr(
        configure_prompts,
        "_import_inquirer_select",
        lambda: FakeInquirer(),
    )

    selected = configure_prompts.select_prompt(
        "Choose model",
        [
            SelectChoice(
                value="gemini-3.8-flash-tiered",
                label="gemini-3.8-flash",
            )
        ],
        default="gemini-3.8-flash-tiered",
    )

    assert selected == "gemini-3.8-flash-tiered"
    assert captured["choices"] == [
        {"name": "• gemini-3.8-flash", "value": "gemini-3.8-flash-tiered"}
    ]
    assert captured["default"] == "gemini-3.8-flash-tiered"


@pytest.mark.parametrize(
    "response",
    (
        "2",
        "gemini-3.8-flash",
        "GEMINI-3.8-FLASH-TIERED",
        None,
    ),
)
def test_click_fallback_returns_canonical_value_without_listing_raw_values(
    monkeypatch, capsys, response: str | None
) -> None:
    prompt_calls = []
    monkeypatch.setattr(configure_prompts, "_import_inquirer_select", lambda: None)

    def fake_prompt(message, **kwargs):
        prompt_calls.append((message, kwargs))
        return kwargs["default"] if response is None else response

    monkeypatch.setattr(configure_prompts.click, "prompt", fake_prompt)

    selected = configure_prompts.select_prompt(
        "Choose model",
        [
            SelectChoice(
                value="gemini-3.7-flash-tiered",
                label="gemini-3.7-flash",
            ),
            SelectChoice(
                value="gemini-3.8-flash-tiered",
                label="gemini-3.8-flash",
            ),
        ],
        default="gemini-3.8-flash-tiered",
    )

    assert selected == "gemini-3.8-flash-tiered"
    assert prompt_calls == [
        (
            "Select option",
            {
                "type": str,
                "default": "2",
                "show_choices": False,
            },
        )
    ]
    output = capsys.readouterr().out
    assert "1. gemini-3.7-flash" in output
    assert "2. gemini-3.8-flash" in output
    assert "gemini-3.7-flash-tiered" not in output
    assert "gemini-3.8-flash-tiered" not in output


def test_gemini_oauth_picker_saves_canonical_id_but_summarizes_display_name(
    monkeypatch,
) -> None:
    config = MobileConfig()
    saved_configs = []
    login_calls = []
    picker_choices = []

    monkeypatch.setattr(configure_wizard.ConfigLoader, "load", lambda: config)
    monkeypatch.setattr(
        configure_wizard.ConfigLoader,
        "save",
        lambda saved, path=None: saved_configs.append(saved),
    )
    monkeypatch.setattr(
        configure_wizard,
        "_oauth_credentials_present",
        lambda credential_path, variant_id: False,
    )

    def select_tiered_model(message, choices, **kwargs):
        assert message == "Choose model"
        picker_choices.extend(choices)
        return next(
            choice.value for choice in choices if choice.label == "gemini-3.8-flash"
        )

    monkeypatch.setattr(configure_wizard, "_select_with_back", select_tiered_model)
    output = StringIO()

    configure_wizard.run_configure_wizard(
        Console(file=output, force_terminal=False, width=120),
        ConfigureWizardCallbacks(
            run_openai_oauth_login=lambda **kwargs: None,
            run_anthropic_oauth_login=lambda **kwargs: None,
            run_gemini_oauth_login=lambda **kwargs: login_calls.append(kwargs),
        ),
        provider="gemini",
        auth_mode="oauth",
        model=None,
        api_key=None,
        base_url=None,
    )

    canonical_id = "gemini-3.8-flash-tiered"
    assert any(
        choice.value == canonical_id and choice.label == "gemini-3.8-flash"
        for choice in picker_choices
    )
    assert saved_configs == [config]
    assert login_calls[0]["model"] == canonical_id
    assert all(
        profile.model == canonical_id for profile in config.llm_profiles.values()
    )
    assert all(
        profile["model"] == canonical_id
        for profile in config.to_dict()["llm_profiles"].values()
    )
    summary = output.getvalue()
    assert "Model: gemini-3.8-flash" in summary
    assert canonical_id not in summary
