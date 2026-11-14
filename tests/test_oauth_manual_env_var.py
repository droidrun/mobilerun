"""Exercise manual-login selection without binding ports or reading real tokens."""

from unittest.mock import Mock, sentinel

import pytest

from mobilerun.agent.utils.oauth import (
    anthropic_oauth_llm,
    gemini_oauth_code_assist_llm,
    openai_oauth_llm,
)
from mobilerun.agent.utils.oauth.login_timeout import OAuthLoginDeadline


class DesktopFlowSelected(Exception):
    """Stop at callback-server creation, before any desktop login side effects."""


@pytest.fixture(params=["anthropic", "gemini", "openai"])
def login(request, monkeypatch, tmp_path):
    monkeypatch.delenv("MOBILERUN_OAUTH_MANUAL", raising=False)
    monkeypatch.delenv("DROIDRUN_OAUTH_MANUAL", raising=False)
    credential_path = str(tmp_path / "auth-profiles.json")
    if request.param == "anthropic":
        module = anthropic_oauth_llm
        cls = module.AnthropicOAuthLLM
        llm = cls(credential_path=credential_path)
        method = "login_headless"
        extra = {"expires_in": 600}
    elif request.param == "gemini":
        module = gemini_oauth_code_assist_llm
        cls = module.GeminiOAuthCodeAssistLLM
        llm = cls(credential_path=credential_path)
        method = "login_headless"
        extra = {"prompt_consent": False, "persist_credentials": False}
    else:
        module = openai_oauth_llm
        cls = module.OpenAIOAuth
        llm = cls(model="gpt-5.5", oauth_credential_path=credential_path)
        method = "_login_device_code"
        extra = {}
        monkeypatch.setattr(module, "_tls_preflight", Mock())

    manual = Mock(return_value=sentinel.credentials)
    server = Mock(side_effect=DesktopFlowSelected)
    headless = Mock(return_value=False)
    monkeypatch.setattr(cls, method, manual)
    monkeypatch.setattr(module, "HTTPServer", server)
    monkeypatch.setattr(module, "_is_headless_environment", headless)
    return llm, manual, server, headless, extra


@pytest.mark.parametrize(
    "current,legacy,expected",
    [
        ("1", None, True),
        ("true", None, True),
        ("yes", None, True),
        ("TRUE", None, True),
        ("Yes", None, True),
        (None, "1", False),
        (None, "true", False),
        (None, "YES", False),
        ("true", "false", True),
        ("false", "true", False),
        ("0", "true", False),
        ("no", "true", False),
        ("off", "true", False),
        ("invalid", "true", False),
        ("", "true", False),
        ("", "false", False),
        ("", None, False),
        ("0", None, False),
        ("false", None, False),
        ("no", None, False),
        ("off", None, False),
        (None, "false", False),
        (None, "", False),
        (None, None, False),
    ],
)
def test_environment_selects_login_flow(login, monkeypatch, current, legacy, expected):
    llm, manual, server, _, _ = login
    if current is not None:
        monkeypatch.setenv("MOBILERUN_OAUTH_MANUAL", current)
    if legacy is not None:
        monkeypatch.setenv("DROIDRUN_OAUTH_MANUAL", legacy)

    if expected:
        assert llm.login(open_browser=False) is sentinel.credentials
        manual.assert_called_once()
        server.assert_not_called()
    else:
        with pytest.raises(DesktopFlowSelected):
            llm.login(open_browser=False)
        server.assert_called_once()
        manual.assert_not_called()


def test_headless_detection_still_selects_manual_with_false_env(login, monkeypatch):
    llm, manual, server, headless, _ = login
    monkeypatch.setenv("MOBILERUN_OAUTH_MANUAL", "false")
    headless.return_value = True

    assert llm.login(open_browser=False) is sentinel.credentials
    manual.assert_called_once()
    server.assert_not_called()


@pytest.mark.parametrize("open_browser", [False, True])
def test_manual_selection_forwards_login_options(login, monkeypatch, open_browser):
    llm, manual, server, _, extra = login
    monkeypatch.setenv("MOBILERUN_OAUTH_MANUAL", "true")
    deadline = OAuthLoginDeadline(30)

    assert (
        llm.login(
            open_browser=open_browser,
            timeout_seconds=17,
            deadline=deadline,
            **extra,
        )
        is sentinel.credentials
    )
    manual.assert_called_once_with(
        open_browser=open_browser,
        timeout_seconds=17,
        deadline=deadline,
        **extra,
    )
    server.assert_not_called()


def test_callback_bind_failure_falls_back_independently_of_env(login, monkeypatch):
    llm, manual, server, _, extra = login
    monkeypatch.setenv("MOBILERUN_OAUTH_MANUAL", "false")
    server.side_effect = OSError("address already in use")
    deadline = OAuthLoginDeadline(30)

    assert (
        llm.login(open_browser=False, timeout_seconds=17, deadline=deadline, **extra)
        is sentinel.credentials
    )
    server.assert_called_once()
    manual.assert_called_once_with(
        open_browser=False, timeout_seconds=17, deadline=deadline, **extra
    )


def test_expired_deadline_does_not_start_either_flow(login, monkeypatch):
    llm, manual, server, _, _ = login
    monkeypatch.setenv("MOBILERUN_OAUTH_MANUAL", "true")
    now = [0.0]
    deadline = OAuthLoginDeadline(1, clock=lambda: now[0])
    now[0] = 2.0

    with pytest.raises(TimeoutError):
        llm.login(open_browser=False, deadline=deadline)
    manual.assert_not_called()
    server.assert_not_called()
