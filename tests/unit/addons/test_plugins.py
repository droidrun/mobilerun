"""Optional plugin discovery, CLI dispatch and base/add-on isolation."""

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from importlib.util import find_spec
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from mobilerun.addons import plugins
from mobilerun.cli.app import _run_agent_macro, cli
from mobilerun.config.user_config import MobileConfig
from tests.support.paths import REPO_ROOT


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

    monkeypatch.setattr(plugins.metadata, "entry_points", lambda *, group: [])
    assert "  agent " not in CliRunner().invoke(cli, ["--help"]).output
    with pytest.raises(
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

    entry = Entry("doctor", RuntimeError("must not load"))
    monkeypatch.setattr(plugins.metadata, "entry_points", lambda *, group: [entry])
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert entry.loads == 0
    assert result.output.count("  doctor ") == 1


def test_removed_alias_name_is_available_to_plugin(monkeypatch):

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

    root = REPO_ROOT / "mobilerun"
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

    from tests.support.macro import write_macro

    received = {}
    operator = Entry("agent_macro", lambda **kwargs: received.update(kwargs))
    original = metadata.entry_points

    def entries(*, group):
        return [operator] if group == "mobilerun.operators" else original(group=group)

    monkeypatch.setattr(plugins.metadata, "entry_points", entries)
    path = write_macro(tmp_path)
    result = CliRunner().invoke(
        cli,
        ["macro", "replay", str(path), "--on-mismatch", "agent", "--android", "fake"],
    )
    assert result.exit_code == 0, result.output
    assert received["path"] == str(path)
    assert operator.loads == 1


def test_fake_agent_entry_point_is_discovered(monkeypatch):
    import types
    from importlib import metadata

    from mobilerun.cli.base import menu

    @click.group()
    def agent():
        pass

    @agent.command()
    @click.argument("task")
    def run(task):
        click.echo(f"run:{task}")

    @agent.group()
    def llm():
        pass

    @llm.command()
    def configure():
        click.echo("configured")

    @llm.group()
    def login():
        pass

    module = types.ModuleType("fake_agent_plugin")
    module.agent = agent
    monkeypatch.setitem(sys.modules, module.__name__, module)
    entry = metadata.EntryPoint(
        name="agent", value="fake_agent_plugin:agent", group="mobilerun.cli"
    )
    original = plugins.metadata.entry_points

    def entries(*args, **kwargs):
        found = list(original(*args, **kwargs))
        if kwargs.get("group") == "mobilerun.cli":
            return [item for item in found if item.name != "agent"] + [entry]
        return found

    monkeypatch.setattr(plugins.metadata, "entry_points", entries)
    runner = CliRunner()
    help_result = runner.invoke(cli, ["--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "  agent " in help_result.output
    assert 'mobilerun agent run "open settings"' in help_result.output

    choices = []
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(
        menu,
        "select_prompt",
        lambda message, items: choices.extend(items) or "__back__",
    )
    result = runner.invoke(cli, [])
    assert result.exit_code == 0, result.output
    assert "agent" in [item.value for item in choices]
    for args, expected in (
        (["agent", "run", "task"], "run:task"),
        (["agent", "llm", "configure"], "configured"),
    ):
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, result.output
        assert expected in result.output


@pytest.mark.parametrize("mode", ["present", "blocked", "absent"])
def test_interactive_menu_uses_available_agent_epilog(mode):
    if mode == "present" and find_spec("mobilerun_agents") is None:
        pytest.skip("present-state test requires the agents add-on")
    script = """
import importlib.abc, sys
from click.testing import CliRunner
from mobilerun.addons import plugins
from mobilerun.cli.base import menu
from mobilerun.cli.app import cli
mode = sys.argv[1]
if mode != "present":
    plugins.discover = lambda group: {}
if mode == "absent":
    sys.modules["mobilerun_agents"] = None
elif mode == "blocked":
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {"mobilerun.agents", "mobilerun_agents"}:
                raise ModuleNotFoundError(name=fullname)
    sys.meta_path.insert(0, Block())
menu.is_interactive = lambda: True
choices = []
def select(message, items):
    choices.extend(item.value for item in items)
    return "__back__"
menu.select_prompt = select
result = CliRunner().invoke(cli, [])
assert result.exit_code == 0, result.output
assert ("agent" in choices) == (mode == "present"), choices
assert ('mobilerun agent run "open settings"' in result.output) == (mode == "present"), result.output
"""
    result = subprocess.run(
        [sys.executable, "-c", script, mode], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("absent", [True, False])
def test_root_visibility_epilog_and_completion_follow_addon(absent):
    if not absent and find_spec("mobilerun_agents") is None:
        pytest.skip("present-state test requires the agents add-on")
    script = """
import importlib.abc
import sys
if sys.argv[1] == "True":
    class BlockAgents(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith(("mobilerun.agents", "mobilerun_agents")):
                raise ModuleNotFoundError("blocked", name=fullname)
    sys.meta_path.insert(0, BlockAgents())
import click
from click.testing import CliRunner
from mobilerun.addons import plugins
if sys.argv[1] == "True":
    plugins.discover = lambda group: {}
from mobilerun.cli.app import cli
result = CliRunner().invoke(cli, ["--help"])
assert result.exit_code == 0, result.output
present = sys.argv[1] == "False"
ctx = click.Context(cli, info_name="mobilerun")
assert ("agent" in cli.list_commands(ctx)) == present
assert ("  agent " in result.output) == present
assert ('mobilerun agent run "open settings"' in result.output) == present
completions = [item.value for item in cli.shell_complete(ctx, "")]
assert ("agent" in completions) == present
assert "run" not in completions and "configure" not in completions
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(absent)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "operation",
    [
        "cli:agent,--help",
        "macro",
    ],
)
def test_missing_agents_is_actionable_in_a_fresh_process(operation):
    script = """
import importlib
import importlib.abc
import sys

class BlockAgents(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("mobilerun.agents", "mobilerun_agents")):
            raise ModuleNotFoundError("blocked", name=fullname)

sys.meta_path.insert(0, BlockAgents())
from mobilerun.addons import plugins
plugins.discover = lambda group: {}
operation = sys.argv[1]
try:
    if operation.startswith("cli:"):
        from click.testing import CliRunner
        from mobilerun.cli.app import cli
        result = CliRunner().invoke(cli, operation[4:].split(","))
        assert result.exit_code != 0
        message = result.output
    elif operation == "macro":
        from mobilerun.cli.app import _run_agent_macro
        _run_agent_macro(None, None)
    elif operation == "export":
        import mobilerun
        mobilerun.MobileAgent
    else:
        importlib.import_module(operation[7:])
except ModuleNotFoundError as error:
    message = str(error)
assert "Install the mobilerun-agents add-on (see README: Agent add-on)." in message, message
assert "pip install" not in message, message
"""
    result = subprocess.run(
        [sys.executable, "-c", script, operation],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_agents_from_another_path_are_discoverable(tmp_path):
    base = tmp_path / "base"
    addon = tmp_path / "addon"
    shutil.copytree(
        REPO_ROOT / "mobilerun",
        base / "mobilerun",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    agents = addon / "mobilerun_agents"
    agents.mkdir(parents=True)
    (agents / "__init__.py").write_text("MARKER = 42\n")
    (addon / "mobilerun" / "agents").mkdir(parents=True)  # Obsolete namespace portion.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(base), str(addon)))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from importlib.util import find_spec; import mobilerun_agents; "
            "assert mobilerun_agents.MARKER == 42; assert find_spec('mobilerun.agents') is None",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_shared_config_round_trips_raw_optional_sections_without_mutating_input():
    data = {
        "_version": 8,
        "device": {"serial": "test"},
        "agent": {"max_steps": 3, "manager": {"stateless": True}},
        "llm_profiles": {"manager": {"provider": "GoogleGenAI", "model": "test"}},
        "mcp": {"enabled": True, "servers": {"local": {"command": "echo"}}},
        "credentials": {"enabled": True, "file_path": "credentials.yaml"},
        "tools": {"disabled_tools": []},
    }
    original = deepcopy(data)
    shared = MobileConfig.from_dict(data)
    for section in ("agent", "llm_profiles", "mcp", "credentials", "tools"):
        assert shared.to_dict()[section] == data[section]
    assert data == original
