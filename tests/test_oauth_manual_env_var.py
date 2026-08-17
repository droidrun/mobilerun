"""Regression tests for the MOBILERUN_OAUTH_MANUAL environment variable.

Every other user-facing env var reads MOBILERUN_* first and falls back to the
legacy DROIDRUN_* name (see ConfigLoader.load and is_telemetry_enabled).
OAUTH_MANUAL was the one that never got the MOBILERUN_* spelling, so a user
following current docs had no way to force the manual/headless login flow.

These tests drive the real ``login()`` branch: ``_is_headless_environment`` is
forced False so the env var is the only thing that can select the manual path,
and the manual entry point is stubbed on the class with a sentinel.
"""

import pytest

from mobilerun.agent.utils.oauth import (
    anthropic_oauth_llm,
    gemini_oauth_code_assist_llm,
    openai_oauth_llm,
)
from mobilerun.agent.utils.oauth.anthropic_oauth_llm import AnthropicOAuthLLM
from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
    GeminiOAuthCodeAssistLLM,
)
from mobilerun.agent.utils.oauth.openai_oauth_llm import OpenAIOAuth

SENTINEL = "manual-flow-selected"

# (module, LLM class, manual-flow method, constructor kwargs).
# The three providers spell their credential-path kwarg differently.
CASES = (
    (
        anthropic_oauth_llm,
        AnthropicOAuthLLM,
        "login_headless",
        {"credential_path": None},
    ),
    (
        gemini_oauth_code_assist_llm,
        GeminiOAuthCodeAssistLLM,
        "login_headless",
        {"credential_path": None},
    ),
    (
        openai_oauth_llm,
        OpenAIOAuth,
        "_login_device_code",
        {"model": "gpt-5.5", "oauth_credential_path": None},
    ),
)

CASE_IDS = [cls.__name__ for _, cls, _, _ in CASES]


@pytest.fixture(autouse=True)
def _clear_oauth_env(monkeypatch):
    monkeypatch.delenv("MOBILERUN_OAUTH_MANUAL", raising=False)
    monkeypatch.delenv("DROIDRUN_OAUTH_MANUAL", raising=False)


def _took_manual_path(monkeypatch, module, cls, manual_method, kwargs) -> bool:
    """Call the real login() and report whether it delegated to the manual flow."""
    monkeypatch.setattr(module, "_is_headless_environment", lambda: False)

    # The OpenAI login() reaches the network before the env-var branch.
    if hasattr(module, "_tls_preflight"):
        monkeypatch.setattr(module, "_tls_preflight", lambda *a, **kw: None)

    reached = []

    def _stub(self, *args, **kwargs):
        reached.append(True)
        return SENTINEL

    monkeypatch.setattr(cls, manual_method, _stub)

    llm = cls(**kwargs)
    try:
        llm.login(open_browser=False, timeout_seconds=0.01)
    except Exception:
        # The interactive desktop path needs a browser, a local callback server
        # and real credentials, so it raises here. Whether the manual branch was
        # taken is recorded above, so this does not mask the result.
        pass
    return bool(reached)


@pytest.mark.parametrize("module,cls,manual_method,kwargs", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("value", ("1", "true", "yes", "TRUE", "Yes"))
def test_mobilerun_oauth_manual_selects_manual_flow(
    monkeypatch, module, cls, manual_method, kwargs, value
):
    monkeypatch.setenv("MOBILERUN_OAUTH_MANUAL", value)
    assert _took_manual_path(monkeypatch, module, cls, manual_method, kwargs)


@pytest.mark.parametrize("module,cls,manual_method,kwargs", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("value", ("1", "true", "yes"))
def test_legacy_droidrun_oauth_manual_still_honoured(
    monkeypatch, module, cls, manual_method, kwargs, value
):
    monkeypatch.setenv("DROIDRUN_OAUTH_MANUAL", value)
    assert _took_manual_path(monkeypatch, module, cls, manual_method, kwargs)


@pytest.mark.parametrize("module,cls,manual_method,kwargs", CASES, ids=CASE_IDS)
def test_mobilerun_takes_precedence_over_legacy(
    monkeypatch, module, cls, manual_method, kwargs
):
    monkeypatch.setenv("MOBILERUN_OAUTH_MANUAL", "true")
    monkeypatch.setenv("DROIDRUN_OAUTH_MANUAL", "false")
    assert _took_manual_path(monkeypatch, module, cls, manual_method, kwargs)


@pytest.mark.parametrize("module,cls,manual_method,kwargs", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("value", ("", "0", "false", "no", "off"))
def test_falsy_values_do_not_select_manual_flow(
    monkeypatch, module, cls, manual_method, kwargs, value
):
    monkeypatch.setenv("MOBILERUN_OAUTH_MANUAL", value)
    assert not _took_manual_path(monkeypatch, module, cls, manual_method, kwargs)


@pytest.mark.parametrize("module,cls,manual_method,kwargs", CASES, ids=CASE_IDS)
def test_unset_does_not_select_manual_flow(
    monkeypatch, module, cls, manual_method, kwargs
):
    assert not _took_manual_path(monkeypatch, module, cls, manual_method, kwargs)
