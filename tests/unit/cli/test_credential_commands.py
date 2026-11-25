"""Account login, credential commands and private file storage."""

import asyncio
import importlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.cli.credentials import group
from mobilerun.cli.devices.base import selection
from mobilerun.clients.cloud import api_keys, device_flow
from mobilerun.clients.local import login_store as credentials
from mobilerun.core.credentials import file as credential_file
from mobilerun.core.credentials.file import FileCredentials
from mobilerun.core.devices.base import Device as DeviceMixin
from mobilerun.errors import CloudAPIError, InvalidAPIKeyError
from mobilerun.models.billing import Billing


class FakeVault:
    def __init__(self, **kwargs):
        pass

    async def close(self):
        pass


def test_account_help_and_commands():
    runner = CliRunner()
    help_result = runner.invoke(cli, ["account", "--help"])
    assert help_result.exit_code == 0
    assert all(name in help_result.output for name in ("login", "logout", "status"))
    root = runner.invoke(cli, ["--help"])
    assert root.exit_code == 0
    assert all(name in root.output for name in ("account", "device", "macro", "doctor"))
    assert ("  agent " in root.output) == (
        importlib.util.find_spec("mobilerun_agents") is not None
    )
    assert "llm" not in root.output
    assert runner.invoke(cli, ["llm", "--help"]).exit_code == 2
    agent = runner.invoke(cli, ["agent", "--help"])
    if importlib.util.find_spec("mobilerun_agents") is None:
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
    result = CliRunner().invoke(cli, ["account", "login", "--billing", "included"])
    assert result.exit_code == 0, result.output
    assert "Logged in as tester@example.test." in result.output
    assert device_flow.request_code.await_args.kwargs["billing"] is Billing.INCLUDED
    assert device_flow.poll_token.await_args.kwargs["billing"] is Billing.INCLUDED
    assert device_flow.fetch_session.await_args.kwargs["billing"] is Billing.INCLUDED


@pytest.mark.parametrize(
    "state,expected",
    [(True, "Logged out."), (False, "Not logged in.")],
    ids=["logged-in", "logged-out"],
)
def test_logout_states(monkeypatch, state, expected):
    monkeypatch.setattr(credentials, "clear_credentials", lambda: state)
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


@pytest.mark.parametrize("option", ["--client-id", "--auth-url"])
def test_api_key_conflicts_with_device_options(option):
    key = "fake-api-key-9876"
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


def test_root_menu_matches_help_without_loading_agent():
    code = """
import sys
import click
from mobilerun.cli.app import cli
from mobilerun.addons.plugins import discover
from mobilerun.addons.settings import CLI_ENTRY_POINT_GROUP
ctx = click.Context(cli)
names = [choice.value for choice in cli.menu_choices(ctx)]
assert names == cli.list_commands(ctx)
assert {"account", "credentials"} <= set(names)
assert ("agent" in names) == ("agent" in discover(CLI_ENTRY_POINT_GROUP))
assert not any(x.startswith(("mobilerun.agents", "mobilerun_agents")) for x in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_vault_list_json_never_outputs_values(monkeypatch):
    class Vault(FakeVault):
        async def list(self, package=None):
            return [{"package": "app", "name": "work", "fields": ["password"]}]

    monkeypatch.setattr(group, "CloudCredentials", Vault)
    result = CliRunner().invoke(cli, ["credentials", "list", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["fields"] == ["password"]


def test_save_json_stdin_and_overwrite_confirmation(monkeypatch):
    calls = []

    class Vault(FakeVault):
        async def list(self, package=None):
            return [{"package": "app", "name": "work", "fields": ["password"]}]

        async def save(self, package, name, fields):
            calls.append(fields)

    monkeypatch.setattr(group, "CloudCredentials", Vault)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["credentials", "save", "app", "work", "--from-json", "-"],
        input='{"password":"private-value"}\nno\n',
    )
    assert not calls
    assert "private-value" not in result.output


def test_save_prompts_hidden_and_delete_confirmation(monkeypatch):
    calls = []

    class Vault(FakeVault):
        async def list(self, package=None):
            return []

        async def save(self, package, name, fields):
            calls.append(fields)

        async def delete(self, *args):
            calls.append(args)

    monkeypatch.setattr(group, "CloudCredentials", Vault)

    def prompt(*args, **kwargs):
        calls.append(kwargs)
        return "private-value"

    monkeypatch.setattr(group.click, "prompt", prompt)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["credentials", "save", "app", "work", "--field", "password"]
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["hide_input"] is True
    assert "private-value" not in result.output
    result = runner.invoke(cli, ["credentials", "delete", "app", "work"], input="n\n")
    assert len(calls) == 2


def test_json_stdin_and_interactive_update_pickers(monkeypatch):
    calls = []

    class Vault(FakeVault):
        async def list(self, package=None):
            return [{"package": "app", "name": "work", "fields": ["password"]}]

        async def save(self, *args):
            calls.append(args)

        async def update(self, *args):
            calls.append(args)

    monkeypatch.setattr(group, "CloudCredentials", Vault)
    monkeypatch.setattr(group, "is_interactive", lambda: True)
    monkeypatch.setattr(group, "select_prompt", lambda label, choices: choices[0].value)
    monkeypatch.setattr(group.click, "prompt", lambda *args, **kwargs: "hidden-value")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["credentials", "save", "app", "new", "--from-json", "-"],
        input='{"password":"hidden-value"}',
    )
    assert result.exit_code == 0, result.output
    assert calls[0] == ("app", "new", {"password": "hidden-value"})
    assert "hidden-value" not in result.output
    result = runner.invoke(cli, ["credentials", "update"])
    assert result.exit_code == 0, result.output
    assert calls[1] == ("app", "work", "password", "hidden-value")
    assert "hidden-value" not in result.output


def test_interactive_delete_offers_field_picker(monkeypatch):
    calls = []

    class Vault(FakeVault):
        async def list(self, package=None):
            return [{"package": "app", "name": "work", "fields": ["password"]}]

        async def delete(self, *args):
            calls.append(args)

    monkeypatch.setattr(group, "CloudCredentials", Vault)
    monkeypatch.setattr(group, "is_interactive", lambda: True)
    monkeypatch.setattr(
        group,
        "select_prompt",
        lambda label, choices: next(c.value for c in choices if c.value == "password"),
    )
    monkeypatch.setattr(group, "confirm_prompt", lambda label, *, default: True)
    result = CliRunner().invoke(cli, ["credentials", "delete", "app", "work"])
    assert result.exit_code == 0, result.output
    assert calls == [("app", "work", "password")]


def test_update_stdin_never_echoes_value(monkeypatch):
    calls = []

    class Vault(FakeVault):
        async def update(self, *args):
            calls.append(args)

    monkeypatch.setattr(group, "CloudCredentials", Vault)
    result = CliRunner().invoke(
        cli,
        ["credentials", "update", "app", "work", "password", "--stdin"],
        input="private-value\n",
    )
    assert result.exit_code == 0, result.output
    assert calls == [("app", "work", "password", "private-value")]
    assert "private-value" not in result.output


def test_credential_confirmations_default_abort_and_yes(monkeypatch):
    calls = []
    prompts = []

    class Vault(FakeVault):
        async def list(self, package=None):
            return [{"package": "app", "name": "work", "fields": ["password"]}]

        async def save(self, *args):
            calls.append(("save", args))

        async def delete(self, *args):
            calls.append(("delete", args))

    monkeypatch.setattr(group, "CloudCredentials", Vault)
    monkeypatch.setattr(group, "is_interactive", lambda: True)
    monkeypatch.setattr(
        group,
        "confirm_prompt",
        lambda message, **kw: prompts.append((message, kw, sys.stdout is sys.stderr))
        or False,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["credentials", "save", "app", "work", "--from-json", "-"],
        input='{"password":"secret"}',
    )
    assert result.exit_code != 0
    result = runner.invoke(cli, ["credentials", "delete", "app", "work", "password"])
    assert result.exit_code != 0
    assert calls == []
    assert len(prompts) == 2
    assert all(item[1] == {"default": False} and item[2] for item in prompts)
    result = runner.invoke(
        cli, ["credentials", "delete", "app", "work", "password", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert len(prompts) == 2
    assert calls[0][0] == "delete"


def test_credential_delete_noninteractive_does_not_read_stdin(monkeypatch):
    monkeypatch.setattr(group, "is_interactive", lambda: False)
    monkeypatch.setattr(
        group,
        "confirm_prompt",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError()),
    )
    result = CliRunner().invoke(
        cli, ["credentials", "delete", "app", "work", "password"], input="y\n"
    )
    assert result.exit_code != 0


def test_mixed_tty_rejects_credential_overwrite_and_delete(monkeypatch):
    calls = []

    class Vault(FakeVault):
        async def list(self, package=None):
            return [{"package": "app", "name": "work", "fields": ["password"]}]

        async def save(self, *args):
            calls.append("save")

        async def delete(self, *args):
            calls.append("delete")

    prompts = []
    monkeypatch.setattr(group, "CloudCredentials", Vault)
    monkeypatch.setattr(group, "is_interactive", lambda: sys.stdout is sys.stderr)
    monkeypatch.setattr(
        group, "confirm_prompt", lambda *args, **kwargs: prompts.append(True) or True
    )
    runner = CliRunner()
    save = runner.invoke(
        cli,
        ["credentials", "save", "app", "work", "--from-json", "-"],
        input='{"password":"secret"}',
    )
    delete = runner.invoke(cli, ["credentials", "delete", "app", "work", "password"])
    assert save.exit_code != 0
    assert delete.exit_code != 0
    assert calls == []
    assert prompts == []


def test_local_file_create_save_preserve_and_restrict(tmp_path):
    path = tmp_path / "private" / "credentials.yaml"
    vault = FileCredentials(path)
    asyncio.run(vault.save("app", "work", {"password": "secret"}))
    asyncio.run(vault.save("other", "home", {"username": "user"}))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert len(asyncio.run(FileCredentials(path).list())) == 2
    asyncio.run(vault.update("app", "work", "password", "new"))
    asyncio.run(vault.delete("other", "home"))
    assert asyncio.run(FileCredentials(path).list()) == [
        {"package": "app", "name": "work", "fields": ["password"]}
    ]


def test_local_file_rejects_loose_permissions(tmp_path):
    path = tmp_path / "credentials.yaml"
    path.write_text("{}")
    path.chmod(0o644)
    with pytest.raises(PermissionError, match="chmod 600"):
        FileCredentials(path)


def test_local_cli_and_merged_list(tmp_path, monkeypatch):
    path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(
        group,
        "get_settings",
        lambda: SimpleNamespace(
            local_credentials_path=path, MOBILERUN_CREDENTIALS_FILE=path
        ),
    )

    class Vault(FakeVault):
        async def list(self, package=None):
            return [{"package": "cloud.app", "name": "cloud", "fields": ["password"]}]

    monkeypatch.setattr(group, "CloudCredentials", Vault)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["credentials", "save", "local.app", "work", "--local", "--from-json", "-"],
        input='{"password":"hidden-value"}',
    )
    assert result.exit_code == 0, result.output
    assert "hidden-value" not in result.output
    result = runner.invoke(cli, ["credentials", "list", "--json"])
    assert result.exit_code == 0, result.output
    assert {item["source"] for item in json.loads(result.output)} == {"vault", "local"}
    assert "hidden-value" not in result.output
    result = runner.invoke(cli, ["credentials", "path"])
    assert str(path) in result.output


def test_required_empty_prompt_cancels(monkeypatch):
    import click

    from mobilerun.cli.base import menu

    @click.command(cls=menu.PromptingCommand)
    @click.argument("value", required=True)
    def cmd(value):
        pass

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "text_prompt", lambda label: "")
    result = CliRunner().invoke(cmd, [])
    assert result.exit_code == 0
    assert "Cancelled." in result.output


def test_credential_table_sorted_and_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(
        group,
        "get_settings",
        lambda: SimpleNamespace(local_credentials_path=tmp_path / "missing"),
    )

    class Vault(FakeVault):
        async def list(self, package=None):
            return [
                {
                    "package": "z.app",
                    "name": "z",
                    "fields": ["password", "username"],
                    "source": "1password",
                },
                {"package": "a.app", "name": "a", "fields": ["email"]},
            ]

    monkeypatch.setattr(group, "CloudCredentials", Vault)
    result = CliRunner().invoke(cli, ["credentials", "list", "--vault"])
    assert result.exit_code == 0, result.output
    assert "Credentials · 2" in result.output
    assert result.output.index("a.app") < result.output.index("z.app")
    assert "username, password" in result.output
    assert "1password" in result.output


def test_cloud_slug_mapping_and_1password_metadata(monkeypatch):
    from types import SimpleNamespace as NS

    from mobilerun.core.credentials import cloud

    item = NS(
        package_name="com-ubercab",
        credential_name="main",
        fields=[NS(field_type="password", value="private")],
        source="onepassword",
    )

    async def listed(*args, **kwargs):
        return [item]

    monkeypatch.setattr(cloud.secrets, "list_secrets", listed)
    monkeypatch.setattr(cloud.secrets, "list_package", listed)

    async def run():
        vault = cloud.CloudCredentials(api_key="fake")
        try:
            first = await vault.list()
            assert first[0]["package"] == "com-ubercab"
            assert first[0]["source"] == "1password"
            assert (await vault.list("com.ubercab"))[0]["package"] == "com.ubercab"
            assert (await vault._find("com.ubercab", "password", "main")) is not None
            assert (await vault.list())[0]["package"] == "com.ubercab"
            assert (await vault.cloud_task_inputs()).credential_references[
                0
            ].package == "com.ubercab"
        finally:
            await vault.close()

    asyncio.run(run())


def test_local_flat_general_update_and_delete(tmp_path):
    path = tmp_path / "credentials.yaml"
    path.write_text("legacy: example\n")
    path.chmod(0o600)
    store = FileCredentials(path)
    asyncio.run(store.update(None, "legacy", "password", "new"))
    assert asyncio.run(store.list()) == [
        {"package": None, "name": "legacy", "fields": ["password"]}
    ]
    asyncio.run(store.delete(None, "legacy"))
    assert asyncio.run(store.list()) == []


def test_interactive_save_picks_store_package_fields_and_default(tmp_path, monkeypatch):
    path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(
        group, "get_settings", lambda: SimpleNamespace(local_credentials_path=path)
    )
    monkeypatch.setattr(group, "is_interactive", lambda: True)
    monkeypatch.setattr(group, "_installed_packages", lambda: ["com.device.app"])
    selections = iter(["local", "com.device.app"])
    monkeypatch.setattr(group, "select_prompt", lambda label, choices: next(selections))
    monkeypatch.setattr(
        group, "multi_select_prompt", lambda label, choices: ["password"]
    )
    monkeypatch.setattr(group, "text_prompt", lambda label, default=None: default)
    monkeypatch.setattr(group.click, "prompt", lambda label, hide_input: "hidden")
    result = CliRunner().invoke(cli, ["credentials", "save"])
    assert result.exit_code == 0, result.output
    assert "hidden" not in result.output
    assert asyncio.run(FileCredentials(path).list()) == [
        {"package": "com.device.app", "name": "default", "fields": ["password"]}
    ]


def test_local_permission_error_shows_chmod_not_contents(tmp_path, monkeypatch):
    path = tmp_path / "credentials.yaml"
    path.write_text("login: private-value")
    path.chmod(0o644)
    monkeypatch.setattr(
        group, "get_settings", lambda: SimpleNamespace(local_credentials_path=path)
    )
    result = CliRunner().invoke(cli, ["credentials", "list", "--local"])
    assert result.exit_code != 0
    assert "chmod 600" in result.output
    assert "private-value" not in result.output


def test_atomic_write_failure_preserves_original(tmp_path, monkeypatch):
    from mobilerun.core.credentials import file as file_module

    path = tmp_path / "credentials.yaml"
    store = FileCredentials(path)
    asyncio.run(store.save("app", "work", {"password": "first"}))
    original = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(file_module.os, "replace", fail_replace)
    with pytest.raises(OSError):
        asyncio.run(store.save("other", "work", {"password": "second"}))
    assert path.read_bytes() == original
    assert list(tmp_path.glob(".credentials-*")) == []


def test_local_update_delete_and_filters(tmp_path, monkeypatch):
    path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(
        group, "get_settings", lambda: SimpleNamespace(local_credentials_path=path)
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["credentials", "save", "app", "name", "--local", "--from-json", "-"],
        input='{"password":"first-private"}',
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        cli,
        ["credentials", "update", "app", "name", "password", "--local", "--stdin"],
        input="second-private\n",
    )
    assert result.exit_code == 0, result.output
    assert "second-private" not in result.output
    result = runner.invoke(cli, ["credentials", "list", "--local", "--json"])
    assert json.loads(result.output) == [
        {"source": "local", "package": "app", "name": "name", "fields": ["password"]}
    ]
    assert "second-private" not in result.output
    result = runner.invoke(
        cli, ["credentials", "delete", "app", "name", "--local", "--yes"]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(cli, ["credentials", "list", "--local"])
    assert "No credentials yet." in result.output


def test_concurrent_file_saves_do_not_lose_entries(tmp_path, monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "credentials.yaml"
    original = FileCredentials._read

    def slow_read(self):
        data = original(self)
        time.sleep(0.05)
        return data

    stores = [FileCredentials(path), FileCredentials(path)]
    monkeypatch.setattr(FileCredentials, "_read", slow_read)

    def save(index):
        asyncio.run(stores[index].save("pkg", f"name{index}", {"password": "secret"}))

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(save, range(2)))
    assert len(asyncio.run(FileCredentials(path).list())) == 2


@pytest.mark.parametrize("private_temp", [False, True])
def test_windows_credentials_require_private_temp(tmp_path, monkeypatch, private_temp):
    windows = SimpleNamespace(**vars(os))
    windows.name = "nt"
    vars(windows).pop("fchmod", None)
    monkeypatch.setattr(credential_file, "os", windows)
    created = []

    def private_file(parent):
        created.append(parent)
        if not private_temp:
            raise PermissionError("owner-only DACL unavailable")
        return credential_file.tempfile.mkstemp(dir=parent)

    monkeypatch.setattr(
        credential_file, "_windows_private_temp", private_file, raising=False
    )
    path = tmp_path / "credentials.yaml"
    store = credential_file.FileCredentials(path)
    if not private_temp:
        with pytest.raises(PermissionError, match="DACL"):
            asyncio.run(store.save("pkg", "login", {"password": "first"}))
        assert created == [tmp_path] and not path.exists()
        return
    asyncio.run(store.save("pkg", "login", {"password": "first"}))
    path.chmod(0o666)  # Windows stat permissions do not represent ACLs.
    asyncio.run(store.update("pkg", "login", "password", "second"))
    assert store._read()["pkg"]["login"] == {"password": "second"}
    assert created == [tmp_path, tmp_path]


@pytest.mark.skipif(os.name != "nt", reason="requires Windows DACL APIs")
def test_native_windows_shared_directory_does_not_expose_temp(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    subprocess.run(
        ["icacls.exe", str(shared), "/grant", "*S-1-1-0:(OI)(CI)R"],
        check=True,
        capture_output=True,
    )
    fd, path = credential_file._windows_private_temp(shared)
    os.close(fd)
    env = dict(os.environ, MOBILERUN_TEST_TEMP=path)
    inspected = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$ErrorActionPreference = 'Stop'; $acl = Get-Acl -LiteralPath $env:MOBILERUN_TEST_TEMP; "
            "$acl.AreAccessRulesProtected; "
            "$acl.GetAccessRules($true,$true,[System.Security.Principal.SecurityIdentifier]) | "
            "ForEach-Object { $_.IdentityReference.Value }",
        ],
        env=env,
        capture_output=True,
        check=True,
        text=True,
    )
    assert inspected.stdout.splitlines() == ["True", "S-1-3-4"]


@pytest.mark.parametrize("configured_path", [True, False])
def test_macro_cli_attaches_config_file_before_replay(
    tmp_path, monkeypatch, configured_path
):
    from mobilerun.macro.cli import macro

    module = importlib.import_module("mobilerun.macro.cli.replay")
    secrets = tmp_path / "credentials.yaml"
    secrets.write_text("login: private-value\n")
    secrets.chmod(0o600)
    path = tmp_path / "macro.json"
    path.write_text(
        json.dumps(
            {
                "macro_schema_version": "2.0",
                "actions": [{"action_type": "type_secret", "secret": "login"}],
            }
        )
    )
    config = SimpleNamespace(
        credentials={
            "enabled": True,
            **({"file_path": str(secrets)} if configured_path else {}),
        },
        device=SimpleNamespace(platform="android", use_tcp=False),
    )
    device = DeviceMixin()
    device.input_text = AsyncMock(return_value=True)
    device._input_text_raw = device.input_text
    device.state.refresh = AsyncMock(
        return_value=SimpleNamespace(
            phone_state={
                "focusedElement": {"className": "EditText"},
                "isEditable": True,
            }
        )
    )
    monkeypatch.setattr(module.ConfigLoader, "load", lambda _: config)
    monkeypatch.setattr(
        module, "get_settings", lambda: SimpleNamespace(local_credentials_path=secrets)
    )
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=device))
    result = CliRunner().invoke(
        macro, ["replay", str(path), "fake-serial", "--delay", "0"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    device.input_text.assert_awaited_once_with("private-value")
    assert "private-value" not in result.output
