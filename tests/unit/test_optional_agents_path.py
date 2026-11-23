"""Optional agents can share the namespace with an editable base install."""

import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest


def test_fake_agent_entry_point_is_discovered(monkeypatch):
    import types
    from importlib import metadata

    import click
    from click.testing import CliRunner

    from mobilerun.addons import plugins
    from mobilerun.cli.app import cli
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
    if mode == "present" and find_spec("mobilerun.agents") is None:
        pytest.skip("present-state test requires the agents add-on")
    script = """
import importlib.abc, sys
import mobilerun
from click.testing import CliRunner
from mobilerun.addons import plugins
from mobilerun.cli.base import menu
from mobilerun.cli.app import cli
mode = sys.argv[1]
if mode != "present":
    plugins.discover = lambda group: {}
if mode == "absent":
    mobilerun.__path__ = []
elif mode == "blocked":
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "mobilerun.agents":
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
    if not absent and find_spec("mobilerun.agents") is None:
        pytest.skip("present-state test requires the agents add-on")
    script = """
import importlib.abc
import sys
if sys.argv[1] == "True":
    class BlockAgents(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "mobilerun.agents" or fullname.startswith("mobilerun.agents."):
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
        if fullname == "mobilerun.agents" or fullname.startswith("mobilerun.agents."):
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
    root = Path(__file__).resolve().parents[2]
    base = tmp_path / "base"
    addon = tmp_path / "addon"
    shutil.copytree(
        root / "mobilerun",
        base / "mobilerun",
        ignore=shutil.ignore_patterns("agents", "__pycache__"),
    )
    agents = addon / "mobilerun" / "agents"
    agents.mkdir(parents=True)
    (agents / "__init__.py").write_text("MARKER = 42\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(base), str(addon)))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mobilerun.agents; " "assert mobilerun.agents.MARKER == 42",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
