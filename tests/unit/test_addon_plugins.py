"""Optional entry points stay lazy and never override built-ins."""

from dataclasses import dataclass
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.app import _run_agent_macro, cli


@dataclass
class Entry:
    name: str
    value: object
    loads: int = 0

    def load(self):
        self.loads += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def test_help_discovers_plugins_without_loading(monkeypatch):
    from mobilerun.addons import plugins

    @click.command()
    def extra():
        click.echo("loaded")

    entry = Entry("extra", extra)
    monkeypatch.setattr(plugins.metadata, "entry_points", lambda *, group: [entry])
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "extra" in result.output
    assert entry.loads == 0
    assert CliRunner().invoke(cli, ["extra"]).output.strip() == "loaded"
    assert entry.loads == 1


def test_missing_and_broken_plugins(monkeypatch):
    from mobilerun.addons import plugins

    monkeypatch.setattr(plugins.metadata, "entry_points", lambda *, group: [])
    assert "  agent " not in CliRunner().invoke(cli, ["--help"]).output
    with __import__("pytest").raises(
        click.ClickException, match="Install the mobilerun-agents add-on"
    ):
        _run_agent_macro(None, None, config_path=None)
    broken = Entry("agent", ValueError("secret-value"))
    monkeypatch.setattr(plugins.metadata, "entry_points", lambda *, group: [broken])
    result = CliRunner().invoke(cli, ["agent"])
    assert result.exit_code != 0
    assert "secret-value" not in result.output
    assert "plugin" in result.output.lower()


@pytest.mark.parametrize("name", ["agent", "agent_macro", "other"])
def test_missing_plugin_message_names_plugin_and_only_agent_has_hint(monkeypatch, name):
    from mobilerun.addons import plugins

    monkeypatch.setattr(plugins, "discover", lambda group: {})
    with pytest.raises(plugins.MissingPlugin) as caught:
        plugins.load("mobilerun.operators", name)
    message = str(caught.value)
    assert f"Command '{name}' is not installed." in message
    assert (
        "Install the mobilerun-agents add-on (see README: Agent add-on)." in message
    ) == (name != "other")
    assert "pip install" not in message


def test_broken_first_party_plugin_names_distribution_without_exception(monkeypatch):
    from mobilerun.addons import plugins

    entry = Entry("macro", ValueError("secret-value"))
    entry.dist = SimpleNamespace(metadata={"Name": "mobilerun"})
    monkeypatch.setattr(plugins, "discover", lambda group: {"macro": entry})
    result = CliRunner().invoke(cli, ["macro"])
    assert (
        "Plugin 'macro' from 'mobilerun' failed to load; reinstall 'mobilerun'."
        in result.output
    )
    assert "secret-value" not in result.output


def test_builtin_wins_over_plugin(monkeypatch):
    from mobilerun.addons import plugins

    entry = Entry("doctor", RuntimeError("must not load"))
    monkeypatch.setattr(plugins.metadata, "entry_points", lambda *, group: [entry])
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert entry.loads == 0
    assert result.output.count("  doctor ") == 1


def test_removed_alias_name_is_available_to_plugin(monkeypatch):
    import json
    from types import SimpleNamespace

    from mobilerun.addons import plugins

    @click.command()
    def run():
        click.echo("plugin run")

    entry = Entry("run", run)
    entry.dist = SimpleNamespace(metadata={"Name": "demo-plugin"}, version="1")
    monkeypatch.setattr(
        plugins.metadata,
        "entry_points",
        lambda *, group: [entry] if group == "mobilerun.cli" else [],
    )
    result = CliRunner().invoke(cli, ["run"])
    assert result.exit_code == 0 and "plugin run" in result.output
    result = CliRunner().invoke(cli, ["plugins", "list", "--json"])
    assert json.loads(result.output)[0]["status"] == "not checked"


def test_replay_handoff_passes_plain_options(monkeypatch):
    from mobilerun.addons import plugins

    received = {}

    def handoff(**kwargs):
        received.update(kwargs)

    entry = Entry("agent_macro", handoff)
    monkeypatch.setattr(plugins.metadata, "entry_points", lambda *, group: [entry])
    _run_agent_macro("openai", "model", config_path="file", path="macro")
    assert received == {
        "provider": "openai",
        "model": "model",
        "config_path": "file",
        "path": "macro",
    }


def test_base_python_has_no_addon_module_paths():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "mobilerun"
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "mobilerun.agents" in (source := path.read_text())
        or (
            path.relative_to(root).as_posix()
            not in {"utils/paths.py", "core/operator/base.py"}
            and "mobilerun_agents" in source
        )
    ]
    assert offenders == []


def test_plugin_list_is_lazy_and_reports_shadowed_and_check_status(monkeypatch):
    import json
    from types import SimpleNamespace

    from mobilerun.addons import plugins

    good = Entry("extra", click.Command("extra"))
    bad = Entry("bad", ValueError("sensitive"))
    shadow = Entry("doctor", click.Command("doctor"))
    operator = Entry("agent_macro", lambda **kwargs: None)
    for entry in (good, bad, shadow, operator):
        entry.dist = SimpleNamespace(metadata={"Name": "demo-plugin"}, version="1.2")

    def entries(*, group):
        return [good, bad, shadow] if group == "mobilerun.cli" else [operator]

    monkeypatch.setattr(plugins.metadata, "entry_points", entries)
    result = CliRunner().invoke(cli, ["plugins", "list", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert {(row["distribution"], row["version"]) for row in rows} == {
        ("demo-plugin", "1.2")
    }
    assert {row["name"]: row["status"] for row in rows} == {
        "extra": "not checked",
        "bad": "not checked",
        "doctor": "shadowed",
        "agent_macro": "not checked",
    }
    assert sum(entry.loads for entry in (good, bad, shadow, operator)) == 0
    result = CliRunner().invoke(cli, ["plugins", "list", "--json", "--check"])
    assert result.exit_code == 0, result.output
    assert {row["name"]: row["status"] for row in json.loads(result.output)} == {
        "extra": "ok",
        "bad": "broken",
        "doctor": "shadowed",
        "agent_macro": "ok",
    }
    assert "sensitive" not in result.output
    assert shadow.loads == 0


def test_first_party_plugin_wins_duplicate_and_list_agrees(monkeypatch):
    import json
    from types import SimpleNamespace

    from mobilerun.addons import plugins

    first = Entry("extra", click.Command("extra"))
    other = Entry("extra", click.Command("other"))
    first.dist = SimpleNamespace(metadata={"Name": "mobilerun"}, version="1")
    other.dist = SimpleNamespace(metadata={"Name": "other"}, version="1")
    monkeypatch.setattr(
        plugins.metadata,
        "entry_points",
        lambda *, group: [other, first] if group == "mobilerun.cli" else [],
    )
    assert plugins.discover("mobilerun.cli")["extra"] is first
    result = CliRunner().invoke(cli, ["plugins", "list", "--json"])
    assert result.exit_code == 0, result.output
    assert {
        row["distribution"]: row["status"] for row in json.loads(result.stdout)
    } == {"mobilerun": "not checked", "other": "shadowed"}


def test_first_party_commands_are_entry_points():
    from mobilerun.addons.plugins import discover
    from mobilerun.addons.settings import CLI_ENTRY_POINT_GROUP

    entries = discover(CLI_ENTRY_POINT_GROUP)
    assert {"macro", "mcp", "skills"} <= entries.keys()
    assert all(
        entries[name].dist.metadata["Name"].lower() == "mobilerun"
        for name in ("macro", "mcp", "skills")
    )


def test_macro_replay_plugin_dispatches_operator_entry_point(monkeypatch, tmp_path):
    from importlib import metadata

    from mobilerun.addons import plugins

    received = {}
    operator = Entry("agent_macro", lambda **kwargs: received.update(kwargs))
    original = metadata.entry_points

    def entries(*, group):
        return [operator] if group == "mobilerun.operators" else original(group=group)

    monkeypatch.setattr(plugins.metadata, "entry_points", entries)
    path = tmp_path / "macro.json"
    path.write_text("{}")
    result = CliRunner().invoke(
        cli,
        ["macro", "replay", str(path), "--on-mismatch", "agent", "--android", "fake"],
    )
    assert result.exit_code == 0, result.output
    assert received["path"] == str(path)
    assert operator.loads == 1
