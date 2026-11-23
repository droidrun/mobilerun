"""The auth group shares the existing cloud device login flow."""

import importlib.util
from types import SimpleNamespace
from unittest.mock import AsyncMock

from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.clients.cloud import api_keys, device_flow
from mobilerun.clients.local import login_store as credentials
from mobilerun.errors import CloudAPIError, InvalidAPIKeyError
from mobilerun.models.billing import Billing


def test_account_help_and_commands():
    runner = CliRunner()
    help_result = runner.invoke(cli, ["account", "--help"])
    assert help_result.exit_code == 0
    assert all(name in help_result.output for name in ("login", "logout", "status"))
    root = runner.invoke(cli, ["--help"])
    assert root.exit_code == 0
    assert all(name in root.output for name in ("account", "device", "macro", "doctor"))
    assert ("  agent " in root.output) == (
        importlib.util.find_spec("mobilerun.agents") is not None
    )
    assert "llm" not in root.output
    assert runner.invoke(cli, ["llm", "--help"]).exit_code == 2
    agent = runner.invoke(cli, ["agent", "--help"])
    if importlib.util.find_spec("mobilerun.agents") is None:
        assert agent.exit_code != 0
        assert (
            "Install the mobilerun-agents add-on (see README: Agent add-on)."
            in agent.output
        )
        return
    assert agent.exit_code == 0
    assert "llm" in agent.output and "run" in agent.output


def test_login_flow(monkeypatch, tmp_path):
    code = SimpleNamespace(
        verification_uri="https://example.test",
        user_code="ABCD",
        verification_uri_complete="",
        expires_in=60,
    )
    monkeypatch.setattr(device_flow, "request_code", AsyncMock(return_value=code))
    monkeypatch.setattr(
        device_flow, "poll_token", AsyncMock(return_value="placeholder")
    )
    monkeypatch.setattr(credentials, "save_token", lambda *a: tmp_path / "auth.json")
    monkeypatch.setattr(
        device_flow,
        "fetch_session",
        AsyncMock(return_value={"user": {"email": "tester@example.test"}}),
    )
    for args in (["account", "login", "--billing", "included"],):
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 0, result.output
        assert "Logged in as tester@example.test." in result.output
    assert device_flow.request_code.await_args.kwargs["billing"] is Billing.INCLUDED
    assert device_flow.poll_token.await_args.kwargs["billing"] is Billing.INCLUDED
    assert device_flow.fetch_session.await_args.kwargs["billing"] is Billing.INCLUDED


def test_logout_states(monkeypatch):
    for state, expected in [(True, "Logged out."), (False, "Not logged in.")]:
        monkeypatch.setattr(credentials, "clear_credentials", lambda state=state: state)
        result = CliRunner().invoke(cli, ["account", "logout"])
        assert result.exit_code == 0
        assert expected in result.output


def test_status_states(monkeypatch):
    monkeypatch.setattr(credentials, "load_credentials", lambda: None)
    result = CliRunner().invoke(cli, ["account", "status"])
    assert result.exit_code != 0
    assert "Not logged in" in result.output
    monkeypatch.setattr(
        credentials, "load_credentials", lambda: {"access_token": "placeholder"}
    )
    monkeypatch.setattr(
        device_flow,
        "fetch_session",
        AsyncMock(return_value={"user": {"email": "tester@example.test", "id": "123"}}),
    )
    result = CliRunner().invoke(cli, ["account", "status"])
    assert result.exit_code == 0
    assert "tester@example.test" in result.output


def test_login_auth_url_option_and_environment(monkeypatch, tmp_path):
    calls = []
    code = SimpleNamespace(
        verification_uri="https://example.test",
        user_code="ABCD",
        verification_uri_complete="",
        expires_in=60,
    )
    monkeypatch.setenv("MOBILERUN_AUTH_URL", "https://env.example.test")
    monkeypatch.setattr(
        device_flow,
        "request_code",
        AsyncMock(
            side_effect=lambda url, client, *, billing: calls.append((url, client))
            or code
        ),
    )
    monkeypatch.setattr(
        device_flow, "poll_token", AsyncMock(return_value="placeholder")
    )
    monkeypatch.setattr(credentials, "save_token", lambda *args: tmp_path / "auth.json")
    monkeypatch.setattr(device_flow, "fetch_session", AsyncMock(return_value=None))
    runner = CliRunner()
    assert runner.invoke(cli, ["account", "login"]).exit_code == 0
    assert (
        runner.invoke(
            cli,
            [
                "account",
                "login",
                "--auth-url",
                "https://option.example.test",
                "--client-id",
                "custom",
            ],
        ).exit_code
        == 0
    )
    assert calls == [
        ("https://env.example.test", "mobilerun-cli"),
        ("https://option.example.test", "custom"),
    ]


def test_api_key_login_validates_and_saves_privately(monkeypatch, tmp_path):
    import json

    key = "fake-api-key-9876"
    path = tmp_path / "credentials.json"
    monkeypatch.setattr(credentials, "credential_path", lambda: path)
    monkeypatch.setattr(api_keys, "validate_api_key", AsyncMock())
    result = CliRunner().invoke(cli, ["account", "login", "--api-key", key])
    assert result.exit_code == 0, result.output
    assert key not in result.output
    assert "fak…9876" in result.output
    assert json.loads(path.read_text())["api_key"] == key
    assert path.stat().st_mode & 0o777 == 0o600


def test_api_key_rejected_or_network_error_does_not_save(monkeypatch, tmp_path):
    path = tmp_path / "credentials.json"
    monkeypatch.setattr(credentials, "credential_path", lambda: path)
    key = "fake-api-key-9876"
    for error in (
        InvalidAPIKeyError("Invalid API key"),
        CloudAPIError("Could not reach the cloud API"),
    ):

        async def fail(key, error=error, *, billing):
            raise error

        monkeypatch.setattr(api_keys, "validate_api_key", fail)
        result = CliRunner().invoke(cli, ["account", "login", "--api-key", key])
        assert result.exit_code == 1
        assert str(error) in result.output
        assert key not in result.output
        assert not path.exists()


def test_api_key_status_logout_and_switching(monkeypatch, tmp_path):
    import json

    from mobilerun.clients.local.login_store import resolve_cloud_api_key

    path = tmp_path / "credentials" / "mobilerun-cloud.json"
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    monkeypatch.setattr(api_keys, "validate_api_key", AsyncMock())
    key = "fake-api-key-9876"
    credentials.save_token("old-token", "https://example.test")
    runner = CliRunner()
    result = runner.invoke(cli, ["account", "login", "--api-key", key])
    assert result.exit_code == 0, result.output
    saved = json.loads(path.read_text())
    assert saved["api_key"] == key and saved["auth_type"] == "api_key"
    assert saved["saved_at"] and "access_token" not in saved and "auth_url" not in saved
    assert resolve_cloud_api_key() == key
    result = runner.invoke(cli, ["account", "status"])
    assert "API key login" in result.output and "fak…9876" in result.output
    assert key not in result.output
    monkeypatch.setenv("MOBILERUN_API_KEY", "env-key-1234")
    result = runner.invoke(cli, ["account", "status"])
    assert "overrides saved credentials" in result.output
    assert "env-key-1234" not in result.output
    result = runner.invoke(cli, ["account", "logout"])
    assert "still set" in result.output and not path.exists()
    assert "env-key-1234" not in result.output
    monkeypatch.delenv("MOBILERUN_API_KEY")
    credentials.save_credentials({"api_key": key, "auth_type": "api_key"})
    credentials.save_token("new-token", "https://example.test")
    assert json.loads(path.read_text()) == {
        "access_token": "new-token",
        "auth_url": "https://example.test",
    }


def test_api_key_conflicts_with_device_options():
    key = "fake-api-key-9876"
    for option in ("--client-id", "--auth-url"):
        result = CliRunner().invoke(
            cli, ["account", "login", "--api-key", key, option, "example"]
        )
        assert result.exit_code != 0
        assert "cannot be combined" in result.output
        assert key not in result.output


def test_login_reports_optional_browser_and_session_failures(monkeypatch, tmp_path):
    import webbrowser
    from importlib import import_module
    from unittest.mock import Mock

    warning = Mock()
    monkeypatch.setattr(
        import_module("mobilerun.cli.auth.login").logger, "warning", warning
    )
    code = SimpleNamespace(
        verification_uri="https://example.test",
        user_code="ABCD",
        verification_uri_complete="https://example.test/verify",
        expires_in=60,
    )
    monkeypatch.setattr(device_flow, "request_code", AsyncMock(return_value=code))
    monkeypatch.setattr(
        device_flow, "poll_token", AsyncMock(return_value="placeholder")
    )
    monkeypatch.setattr(credentials, "save_token", lambda *args: tmp_path / "auth.json")
    monkeypatch.setattr(
        device_flow,
        "fetch_session",
        AsyncMock(side_effect=CloudAPIError("unavailable")),
    )
    monkeypatch.setattr(
        webbrowser, "open", lambda url: (_ for _ in ()).throw(OSError("unavailable"))
    )
    result = CliRunner().invoke(cli, ["account", "login"])
    assert result.exit_code == 0
    assert warning.call_count == 2
