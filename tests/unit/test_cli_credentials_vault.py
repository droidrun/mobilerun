import json
import subprocess
import sys

from click.testing import CliRunner

from mobilerun.cli.app import cli


def test_root_menu_matches_help_without_loading_agent():
    code = """
import sys
import click
from mobilerun.cli.app import cli
from mobilerun.addons.agents import agents_available
ctx = click.Context(cli)
names = [choice.value for choice in cli.menu_choices(ctx)]
assert names == cli.list_commands(ctx)
assert {"account", "credentials"} <= set(names)
assert ("agent" in names) == agents_available()
assert not any(x.startswith("mobilerun.agents") for x in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_vault_list_json_never_outputs_values(monkeypatch):
    from mobilerun.cli.credentials import group

    class Vault:
        def __init__(self, **kwargs):
            pass

        async def list(self, package=None):
            return [{"package": "app", "name": "work", "fields": ["password"]}]

        async def close(self):
            pass

    monkeypatch.setattr(group, "CloudCredentials", Vault)
    result = CliRunner().invoke(cli, ["credentials", "list", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["fields"] == ["password"]


def test_save_json_stdin_and_overwrite_confirmation(monkeypatch):
    from mobilerun.cli.credentials import group

    calls = []

    class Vault:
        def __init__(self, **kwargs):
            pass

        async def list(self, package=None):
            return [{"package": "app", "name": "work", "fields": ["password"]}]

        async def save(self, package, name, fields):
            calls.append(fields)

        async def close(self):
            pass

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
    from mobilerun.cli.credentials import group

    calls = []

    class Vault:
        def __init__(self, **kwargs):
            pass

        async def list(self, package=None):
            return []

        async def save(self, package, name, fields):
            calls.append(fields)

        async def delete(self, *args):
            calls.append(args)

        async def close(self):
            pass

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
    from mobilerun.cli.credentials import group

    calls = []

    class Vault:
        def __init__(self, **kwargs):
            pass

        async def list(self, package=None):
            return [{"package": "app", "name": "work", "fields": ["password"]}]

        async def save(self, *args):
            calls.append(args)

        async def update(self, *args):
            calls.append(args)

        async def close(self):
            pass

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
    from mobilerun.cli.credentials import group

    calls = []

    class Vault:
        def __init__(self, **kwargs):
            pass

        async def list(self, package=None):
            return [{"package": "app", "name": "work", "fields": ["password"]}]

        async def delete(self, *args):
            calls.append(args)

        async def close(self):
            pass

    monkeypatch.setattr(group, "CloudCredentials", Vault)
    monkeypatch.setattr(group, "is_interactive", lambda: True)
    monkeypatch.setattr(
        group,
        "select_prompt",
        lambda label, choices: next(c.value for c in choices if c.value == "password"),
    )
    result = CliRunner().invoke(cli, ["credentials", "delete", "app", "work", "--yes"])
    assert result.exit_code == 0, result.output
    assert calls == [("app", "work", "password")]


def test_update_stdin_never_echoes_value(monkeypatch):
    from mobilerun.cli.credentials import group

    calls = []

    class Vault:
        def __init__(self, **kwargs):
            pass

        async def update(self, *args):
            calls.append(args)

        async def close(self):
            pass

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
    from mobilerun.cli.credentials import group

    calls = []
    prompts = []

    class Vault:
        def __init__(self, **kwargs):
            pass

        async def list(self, package=None):
            return [{"package": "app", "name": "work", "fields": ["password"]}]

        async def save(self, *args):
            calls.append(("save", args))

        async def delete(self, *args):
            calls.append(("delete", args))

        async def close(self):
            pass

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
    from mobilerun.cli.credentials import group

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
    from mobilerun.cli.credentials import group

    calls = []

    class Vault:
        def __init__(self, **kwargs):
            pass

        async def list(self, package=None):
            return [{"package": "app", "name": "work", "fields": ["password"]}]

        async def save(self, *args):
            calls.append("save")

        async def delete(self, *args):
            calls.append("delete")

        async def close(self):
            pass

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
