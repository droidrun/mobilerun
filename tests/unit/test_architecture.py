"""Feature packages are removable and live outside shared layers."""

import ast
import subprocess
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest

from tests.feature_packages import FEATURES

ROOT = Path(__file__).resolve().parents[2] / "mobilerun"
SHARED = ("errors", "models", "utils", "cli", "config", "logs", "core", "clients")


def test_base_contains_no_agent_module_references():
    allowed = {Path("utils/paths.py"), Path("core/operator/base.py")}
    violations = []
    for path in ROOT.rglob("*.py"):
        source = path.read_text()
        if "mobilerun.agents" in source or (
            path.relative_to(ROOT) not in allowed and "mobilerun_agents" in source
        ):
            violations.append(str(path))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                package = ".".join(("mobilerun", *path.relative_to(ROOT).parent.parts))
                module = node.module or ""
                if node.level:
                    module = resolve_name("." * node.level + module, package)
                if module == "mobilerun" and any(
                    a.name == "agents" for a in node.names
                ):
                    violations.append(f"{path}:{node.lineno}")
    assert not violations, violations


@pytest.mark.parametrize("feature", FEATURES)
def test_feature_is_removable(feature):
    script = """
import importlib.abc, sys
from importlib import metadata
feature = sys.argv[1]
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mobilerun." + feature or fullname.startswith("mobilerun." + feature + "."):
            raise ImportError("blocked feature " + fullname)
sys.meta_path.insert(0, Block())
original = metadata.entry_points
def filtered(*args, **kwargs):
    return [entry for entry in original(*args, **kwargs) if not (entry.group == "mobilerun.cli" and entry.name == feature)]
metadata.entry_points = filtered
import mobilerun
import mobilerun.core
from mobilerun.cli.app import cli
from click.testing import CliRunner
result = CliRunner().invoke(cli, ["--help"])
assert result.exit_code == 0, (result.output, result.exception)
assert feature not in [line.strip().split()[0] for line in result.output.splitlines() if line.startswith("  ") and line.strip()]
"""
    result = subprocess.run(
        [sys.executable, "-c", script, feature], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_shared_layers_do_not_contain_or_import_features():
    violations = []
    for folder in SHARED:
        for path in (ROOT / folder).rglob("*.py"):
            if any(
                part in FEATURES
                for part in path.relative_to(ROOT / folder).with_suffix("").parts
            ):
                violations.append(str(path))
            package = ".".join(("mobilerun", *path.relative_to(ROOT).parent.parts))
            for node in ast.walk(ast.parse(path.read_text())):
                targets = []
                if isinstance(node, ast.Import):
                    targets = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level:
                        module = resolve_name("." * node.level + module, package)
                    targets = [
                        module,
                        *(module + "." + alias.name for alias in node.names),
                    ]
                elif (
                    isinstance(node, ast.Call)
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    func = node.func
                    if (
                        isinstance(func, ast.Name)
                        and func.id in {"import_module", "__import__", "find_spec"}
                    ) or (
                        isinstance(func, ast.Attribute)
                        and func.attr in {"import_module", "find_spec"}
                    ):
                        targets = [node.args[0].value]
                for target in targets:
                    if any(
                        target == "mobilerun." + feature
                        or target.startswith("mobilerun." + feature + ".")
                        for feature in FEATURES
                    ):
                        violations.append(f"{path}:{node.lineno}: {target}")
    assert not violations, violations
