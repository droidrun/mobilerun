import asyncio
import stat
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.core.credentials.file import FileCredentials


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
    from mobilerun.cli.credentials import group

    path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(
        group, "get_settings", lambda: SimpleNamespace(local_credentials_path=path)
    )

    class Vault:
        def __init__(self, **kwargs):
            pass

        async def list(self, package=None):
            return [{"package": "cloud.app", "name": "cloud", "fields": ["password"]}]

        async def close(self):
            pass

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
    import json

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
    from mobilerun.cli.credentials import group

    monkeypatch.setattr(
        group,
        "get_settings",
        lambda: SimpleNamespace(local_credentials_path=tmp_path / "missing"),
    )

    class Vault:
        def __init__(self, **kwargs):
            pass

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

        async def close(self):
            pass

    monkeypatch.setattr(group, "CloudCredentials", Vault)
    result = CliRunner().invoke(cli, ["credentials", "list", "--vault"])
    assert result.exit_code == 0, result.output
    assert "Credentials · 2" in result.output
    assert result.output.index("a.app") < result.output.index("z.app")
    assert "username, password" in result.output
    assert "1password" in result.output


def test_agent_default_local_path(monkeypatch, tmp_path):
    pytest.importorskip(
        "mobilerun.agents", reason="requires the optional mobilerun-agents add-on"
    )
    from mobilerun.agents.config import user_config
    from mobilerun.agents.config.user_config import CredentialsConfig

    monkeypatch.setattr(
        user_config,
        "get_settings",
        lambda: SimpleNamespace(local_credentials_path=tmp_path / "vault.yaml"),
    )
    assert CredentialsConfig().file_path == str(tmp_path / "vault.yaml")
    assert CredentialsConfig(file_path="relative.yaml").file_path == "relative.yaml"


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
    from mobilerun.cli.credentials import group

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
    from mobilerun.cli.credentials import group

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
    import json

    from mobilerun.cli.credentials import group

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
