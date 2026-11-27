"""First-party features can be removed without changing CLI metadata."""

import subprocess
import sys
from importlib.metadata import EntryPoint
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from mobilerun.addons import plugins
from mobilerun.cli.app import cli
from tests.feature_packages import FEATURES
from tests.support.paths import REPO_ROOT


@pytest.mark.parametrize("feature", FEATURES)
def test_feature_folder_is_removable_with_entry_point_registered(feature):
    script = """
import importlib.abc
import sys
from importlib import import_module, metadata
from pathlib import Path

from click.testing import CliRunner

feature = sys.argv[1]
package = f"mobilerun.{feature}"

def is_feature(module):
    return module == package or module.startswith(package + ".")

class BlockFeature(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if is_feature(fullname):
            raise ModuleNotFoundError(name=fullname)

assert any(
    entry.name == feature and is_feature(entry.module)
    for entry in metadata.entry_points(group="mobilerun.cli")
)
sys.meta_path.insert(0, BlockFeature())

from mobilerun.cli.app import cli

runner = CliRunner()
result = runner.invoke(cli, ["--help"], prog_name="mobilerun")
assert result.exit_code == 0, result.output
commands = {line.split()[0] for line in result.output.splitlines() if line.startswith("  ") and line.strip()}
assert feature not in commands, result.output
assert "mobilerun " + feature + " " not in result.output, result.output
assert {"device", "sandbox", "mcp", "macro", "skills"} - {feature} <= commands
example = "mobilerun device --help" if feature == "skills" else "mobilerun skills install"
assert example in result.output, result.output

result = runner.invoke(cli, [feature, "--help"], prog_name="mobilerun")
assert f"No such command '{feature}'" in result.output, result.output

for path in sorted(Path("mobilerun").rglob("*.py")):
    module_path = path.parent if path.stem == "__init__" else path.with_suffix("")
    module = ".".join(module_path.parts)
    if not is_feature(module):
        import_module(module)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, feature],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_malformed_third_party_entry_point_does_not_break_root_help(monkeypatch):
    class MalformedEntryPoint(EntryPoint):
        @property
        def module(self):
            raise ValueError("malformed plugin target")

    entry = MalformedEntryPoint(
        name="third-party", value="invalid target:command", group="mobilerun.cli"
    )._for(SimpleNamespace(metadata={"Name": "third-party-plugin"}))
    monkeypatch.setattr(plugins.metadata, "entry_points", lambda *, group: [entry])

    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0, result.exception
    assert "  third-party " in result.output
    result = runner.invoke(cli, ["third-party"])
    assert result.exit_code != 0
    assert (
        "Plugin 'third-party' from 'third-party-plugin' failed to load" in result.output
    )


@pytest.mark.parametrize(
    ("target", "error_type"),
    [
        ("mobilerun.macro.cli.group", "ImportError"),
        ("mobilerun.macro.cli.group", "ModuleNotFoundError"),
        ("mobilerun.macro", "ModuleNotFoundError"),
    ],
)
def test_installed_feature_import_error_is_reported(target, error_type):
    script = """
import importlib.abc
import sys
from importlib.util import spec_from_loader

from click.testing import CliRunner

from mobilerun.cli.app import cli

broken_module, error_type = sys.argv[1:]

class BrokenModule(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if error_type == "ImportError":
            raise ImportError("broken import")
        raise ModuleNotFoundError(name="mobilerun.macro.missing_dependency")

class BreakMacro(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == broken_module:
            return spec_from_loader(fullname, BrokenModule(), is_package=fullname == "mobilerun.macro")

sys.meta_path.insert(0, BreakMacro())
runner = CliRunner()
help_result = runner.invoke(cli, ["--help"])
assert help_result.exit_code == 0, help_result.output
assert "  macro " in help_result.output, help_result.output
result = runner.invoke(cli, ["macro", "--help"])
assert result.exit_code != 0, result.output
assert "Plugin 'macro' from 'mobilerun' failed to load" in result.output, result.output
"""
    result = subprocess.run(
        [sys.executable, "-c", script, target, error_type],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
