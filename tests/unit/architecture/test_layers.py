"""Package layout, import boundaries and optional-feature isolation contracts."""

import ast
import os
import subprocess
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest

from tests.feature_packages import FEATURES
from tests.support.paths import REPO_ROOT

ROOT = REPO_ROOT / "mobilerun"
LOCAL = ROOT / "clients" / "local"


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


def test_device_package_does_not_import_cloud_driver_eagerly():
    script = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'mobilerun_core_local.driver.cloud':
        raise ImportError('cloud extra unavailable')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import mobilerun.core.devices as devices
assert devices.AndroidDevice
try:
    devices.CloudDevice
except ImportError as error:
    assert 'cloud extra unavailable' in str(error)
else:
    raise AssertionError('CloudDevice should load the optional driver on access')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_sdk_imports_only_live_in_cloud_clients():
    violations = []
    for path in ROOT.rglob("*.py"):
        if path.is_relative_to(ROOT / "clients" / "cloud"):
            continue
        for line, target in _references(path):
            if (
                path == ROOT / "core/devices/components/identifiers.py"
                and target == "mobilerun_sdk"
                and path.read_text().splitlines()[line - 1].strip()
                == '"mobilerun_sdk",'
            ):
                continue  # Logger name, not an import or entry point.
            if target == "mobilerun_sdk" or target.startswith("mobilerun_sdk."):
                if path == ROOT / "sandbox" / "clients.py":
                    continue  # Sandbox is a self-contained plugin with its own adapter.
                violations.append(f"{path.relative_to(ROOT)}:{line}")
    assert not violations, violations


def test_clients_and_utils_do_not_import_core():
    for folder in ("clients", "utils"):
        for path in (ROOT / folder).rglob("*.py"):
            for line, target in _references(path):
                assert not target.startswith("mobilerun.core"), f"{path}:{line}"


def test_clients_do_not_import_root_package():
    for path in (ROOT / "clients").rglob("*.py"):
        for line, target in _agent_import_references(path):
            assert target != "mobilerun", f"{path}:{line}"


def test_root_import_scanner_detects_both_import_forms(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("import mobilerun\nfrom mobilerun import __version__\n")
    assert [target for _, target in _agent_import_references(path)].count(
        "mobilerun"
    ) == 2


def test_utils_import_only_utils_from_mobilerun():
    """Standalone utilities must not depend on domain packages."""
    violations = []
    for path in (ROOT / "utils").rglob("*.py"):
        for line, target in _references(path):
            if path == ROOT / "utils/screen_fingerprint.py" and target in {
                "mobilerun.models.recording",
                "mobilerun.models.recording.Fingerprint",
            }:
                continue  # Pure run evidence value shared by core and replay.
            if target.startswith("mobilerun.") and not (
                target == "mobilerun.utils" or target.startswith("mobilerun.utils.")
            ):
                violations.append(f"{path.relative_to(ROOT)}:{line}: {target}")
    assert not violations, violations


def test_image_sizing_accepts_an_explicit_limit():
    from mobilerun.utils.images import fit_dimensions_to_max_side

    assert fit_dimensions_to_max_side(3000, 1500, 1000) == (1000, 500)


def _references(path):
    package = ".".join(("mobilerun", *path.relative_to(ROOT).parent.parts))
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = resolve_name("." * node.level + module, package)
            yield node.lineno, module
            for alias in node.names:
                yield node.lineno, module + "." + alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Covers import_module/__import__ and configured entry-point targets.
            yield node.lineno, node.value


def _agent_import_references(path):
    package = (
        ".".join(("mobilerun", *path.relative_to(ROOT).parent.parts))
        if path.is_relative_to(ROOT)
        else "mobilerun"
    )
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = resolve_name("." * node.level + module, package)
            yield node.lineno, module
            for alias in node.names:
                yield node.lineno, module + "." + alias.name
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "update"
                and isinstance(func.value, ast.Name)
                and func.value.id in {"LEGACY_MODULES", "_LEGACY_EXPORTS"}
            ):
                for value in ast.walk(node.args[0]):
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        yield value.lineno, value.value
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else ""
            )
            arg = node.args[0]
            if (
                name in {"import_module", "find_spec", "__import__"}
                and isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
            ):
                yield node.lineno, arg.value
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in {"LEGACY_MODULES", "_LEGACY_EXPORTS"}
            for target in node.targets
        ):
            for value in ast.walk(node.value):
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    yield value.lineno, value.value


def test_agent_boundary_scanner_distinguishes_imports_from_strings(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("""
message = "mobilerun_agents"
import mobilerun_agents.agent.operator
from mobilerun_agents import AgentOperator
import_module("mobilerun_agents.cli")
find_spec("mobilerun_agents")
__import__("mobilerun_agents.agent.tools")
LEGACY_MODULES = {"old": "mobilerun_agents.config"}
LEGACY_MODULES.update({"other": "mobilerun_agents.macro_agent"})
""")
    targets = [target for _, target in _agent_import_references(path)]
    assert targets.count("mobilerun_agents") == 2
    assert "mobilerun_agents.agent.operator" in targets
    assert "mobilerun_agents.cli" in targets
    assert "mobilerun_agents.agent.tools" in targets
    assert "mobilerun_agents.config" in targets
    assert "mobilerun_agents.macro_agent" in targets


def test_base_does_not_import_agents():
    violations = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        for line, target in _agent_import_references(path):
            if (
                relative == Path("utils/paths.py")
                and target == "mobilerun_agents"
                and path.read_text().splitlines()[line - 1].strip()
                == 'spec = find_spec("mobilerun_agents")'
            ):
                continue  # Read-only legacy asset lookup, not an import.
            if target.startswith(("mobilerun.agents", "mobilerun_agents")):
                violations.append(f"{relative}:{line}")
    assert not violations, violations


def test_shared_packages_do_not_import_features():
    violations = []
    for folder in (
        "core",
        "clients",
        "config",
        "utils",
        "errors",
        "logs",
        "models",
        "cli",
    ):
        for path in (ROOT / folder).rglob("*.py"):
            for line, target in _agent_import_references(path):
                if any(
                    target == f"mobilerun.{name}"
                    or target.startswith(f"mobilerun.{name}.")
                    for name in FEATURES
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{line}: {target}")
    assert not violations, violations


def test_independent_features_do_not_import_each_other_or_agents():
    violations = []
    for feature in FEATURES:
        forbidden = (set(FEATURES) | {"agents"}) - {feature}
        for path in (ROOT / feature).rglob("*.py"):
            for line, target in _references(path):
                if "mobilerun_agents" in target or any(
                    "mobilerun." + other in target for other in forbidden
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{line}: {target}")
    assert not violations, violations


def test_macro_dependencies_are_shared_or_internal():
    allowed = {
        "models",
        "macro",
        "core",
        "clients",
        "config",
        "utils",
        "logs",
        "errors",
    }
    for path in (ROOT / "macro").rglob("*.py"):
        for line, target in _references(path):
            if target.startswith("mobilerun."):
                assert (
                    target.split(".")[1] in allowed
                    or target == "mobilerun.cli.base"
                    or target.startswith("mobilerun.cli.base.")
                    or target == "mobilerun.cli.devices.base"
                    or target.startswith("mobilerun.cli.devices.base.")
                ), f"{path}:{line}: {target}"


def test_features_only_import_shared_cli_base():
    for feature in FEATURES:
        for path in (ROOT / feature).rglob("*.py"):
            for line, target in _references(path):
                if target == "mobilerun.cli" or target.startswith("mobilerun.cli."):
                    assert any(
                        target == base or target.startswith(base + ".")
                        for base in ("mobilerun.cli.base", "mobilerun.cli.devices.base")
                    ), f"{path}:{line}: {target}"


@pytest.mark.parametrize("module", ["mobilerun", "mobilerun.macro"])
def test_import_is_agent_free(module):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, sys; importlib.import_module(sys.argv[1]); "
            "assert not any(m.startswith(('mobilerun.agents', 'mobilerun_agents')) for m in sys.modules)",
            module,
        ],
        check=True,
    )


def test_shared_packages_do_not_import_cli_or_click():
    for folder in ("clients", "core", "config", "utils"):
        for path in (ROOT / folder).rglob("*.py"):
            for _line, target in _references(path):
                assert not (
                    target == "click"
                    or target.startswith("click.")
                    or target == "mobilerun.cli"
                    or target.startswith("mobilerun.cli.")
                ), path


def test_root_cli_help_does_not_load_agents():
    script = """
from click.testing import CliRunner
from mobilerun.addons.plugins import discover
from mobilerun.addons.settings import CLI_ENTRY_POINT_GROUP
from mobilerun.cli.app import cli
import sys
result = CliRunner().invoke(cli, ["--help"])
assert result.exit_code == 0, result.output
names = {"account", "device", "doctor", "macro", "skills"}
if ("agent" in discover(CLI_ENTRY_POINT_GROUP)):
    names.add("agent")
rows = [line.strip().split(maxsplit=1) for line in result.output.splitlines()
        if line.startswith("  ") and line.strip().split()[0] in names | {"agent"}]
assert {row[0] for row in rows} == names, rows
assert all(len(row) == 2 and row[1] for row in rows), rows
assert not any(m.startswith(("mobilerun.agents", "mobilerun_agents")) for m in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_macro_help_and_agent_dry_run_do_not_load_agents(tmp_path):
    from tests.support.macro import write_macro

    macro = write_macro(tmp_path)
    script = """
import sys
from click.testing import CliRunner
from mobilerun.cli.app import cli
runner = CliRunner()
for args in (["macro", "--help"], ["macro", "replay", "--help"],
             ["macro", "replay", sys.argv[1], "--on-mismatch", "agent", "--dry-run"]):
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert not any(name.startswith(("mobilerun.agents", "mobilerun_agents")) for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(macro)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_state_public_exports_are_light_and_old_paths_are_gone():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import mobilerun.core.state as state; from importlib.util import find_spec; assert state.UIElement and state.UIState and state.StateProvider; assert state.AndroidStateProvider and state.IOSStateProvider and state.ScreenshotOnlyStateProvider; assert all(find_spec('mobilerun.core.state.' + name) is None for name in ['provider', 'ios_provider', 'screenshot_provider', 'state', 'stealth_state'])",
        ],
        check=True,
    )


def test_models_import_only_standard_library_third_party_and_errors():
    for path in (ROOT / "models").rglob("*.py"):
        for line, target in _references(path):
            if target.startswith("mobilerun."):
                assert target.startswith(
                    ("mobilerun.models", "mobilerun.errors")
                ), f"{path}:{line}: {target}"


def test_resource_and_device_folder_contracts():
    assert not (ROOT / "testing").exists(), "Test support must not ship in mobilerun"
    support = REPO_ROOT / "tests" / "support"
    for name in ("__init__", "devices", "extensions"):
        assert (support / f"{name}.py").is_file()
    for name in ("phone_number", "mailbox", "proxy", "credentials"):
        folder = ROOT / "core" / name
        assert (folder / "base.py").is_file()
        assert (folder / "README.md").is_file()
    devices = ROOT / "core" / "devices"
    for name in (
        "android",
        "ios",
        "cloud",
        "remote",
        "base",
        "pool",
        "recording",
        "stealth",
    ):
        assert (devices / f"{name}.py").is_file()
    for name in ("cloud_apps", "cloud_lifecycle", "wrapper"):
        assert (devices / "components" / f"{name}.py").is_file()
    assert (ROOT / "core" / "operator" / "cloud.py").is_file()
    for name in ("base", "android", "ios", "cloud", "remote", "wrappers"):
        assert (
            (devices / f"{name}.py")
            if name != "wrappers"
            else (devices / "components/wrapper.py")
        ).is_file()
    assert (devices / "README.md").is_file()
    assert sorted(path.name for path in devices.glob("*.md")) == ["README.md"]
    assert not any((ROOT / "core" / "comms").glob("*.py"))


def test_local_client_layout_and_import_boundaries():
    expected = {
        "apk.py",
        "emulator.py",
        "login_store.py",
    }
    assert expected <= {path.name for path in LOCAL.glob("*.py")}
    # File credentials are a core provider, not a local-client adapter.
    assert (ROOT / "core" / "credentials" / "file.py").is_file()
    for path in LOCAL.glob("*.py"):
        tree = ast.parse(path.read_text())
        assert not any(isinstance(node, ast.ClassDef) for node in ast.walk(tree)), path
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                modules += [f"{node.module}.{alias.name}" for alias in node.names]
            assert not any(
                module == prefix or module.startswith(prefix + ".")
                for module in modules
                for prefix in ("mobilerun.core", "mobilerun.clients.cloud")
            ), path


def test_cli_domains_have_readmes_and_no_common_package():
    cli = ROOT / "cli"
    assert not (cli / "common").exists()
    for domain in ("auth", "base", "devices", "phone_number", "mailbox", "proxy"):
        assert (cli / domain / "README.md").is_file(), domain
    assert not (cli / "comms.py").exists()


# Selected initializers expose lazy public names without importing optional dependencies.
INITIALIZER_EXCEPTIONS = {
    "__init__.py": "Lazy public exports",
    "core/__init__.py": "Lazy public core exports",
    "core/devices/__init__.py": "Lazy public device exports",
    "sandbox/__init__.py": "Actionable error when the optional routines dependency is missing",
}


def test_initializers_only_export_names():
    violations = []
    for path in sorted(ROOT.rglob("__init__.py")):
        # CLI subgroup packages intentionally define their Click group at the
        # package entry point; command implementations live in sibling modules.
        if path.parent.relative_to(ROOT).as_posix() in {
            f"cli/devices/{name}"
            for name in (
                "app",
                "file",
                "input",
                "screen",
                "settings",
                "action",
                "cloud",
                "settings/location",
                "settings/language",
                "settings/timezone",
                "input/clipboard",
            )
        }:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for index, node in enumerate(tree.body):
            relative = path.relative_to(ROOT).as_posix()
            if relative in INITIALIZER_EXCEPTIONS and (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id
                in {
                    "__init__.py": {
                        "__all__",
                        "_EAGER_EXPORTS",
                        "_CORE_EXPORTS",
                    },
                    "core/__init__.py": {"_EXPORT_MODULES"},
                    "core/devices/__init__.py": {"_EXPORT_MODULES"},
                    "sandbox/__init__.py": set(),
                }[relative]
                or isinstance(node, ast.FunctionDef)
                and relative != "sandbox/__init__.py"
                and node.name in {"__getattr__", "__dir__"}
                or relative == "sandbox/__init__.py"
                and isinstance(node, ast.Try)
            ):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
                and not node.orelse
                and all(
                    isinstance(item, (ast.Import, ast.ImportFrom)) for item in node.body
                )
            ):
                # Type-only exports keep optional dependencies lazy at runtime.
                continue
            if (
                index == 0
                and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            if isinstance(node, ast.Assign) and all(
                isinstance(target, ast.Name)
                and (target.id == "__all__" or target.id.startswith("_"))
                for target in node.targets
            ):
                try:
                    value = ast.literal_eval(node.value)
                    target = node.targets[0].id
                    if (
                        target == "__all__"
                        and isinstance(value, (list, tuple))
                        and all(isinstance(item, str) for item in value)
                    ):
                        continue
                    if (
                        target != "__all__"
                        and isinstance(value, dict)
                        and all(
                            isinstance(key, str)
                            and isinstance(item, tuple)
                            and len(item) == 2
                            and all(isinstance(part, str) for part in item)
                            for key, item in value.items()
                        )
                    ):
                        continue
                except (ValueError, TypeError):
                    pass
            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                allowed_calls = {"getattr", "import_module", "AttributeError"}
                statements = (
                    ast.FunctionDef,
                    ast.If,
                    ast.Assign,
                    ast.Return,
                    ast.Raise,
                    ast.ImportFrom,
                )
                if (
                    all(
                        isinstance(stmt, statements)
                        for stmt in ast.walk(node)
                        if isinstance(stmt, ast.stmt)
                    )
                    and all(
                        isinstance(assignment.value, ast.Subscript)
                        for assignment in ast.walk(node)
                        if isinstance(assignment, ast.Assign)
                    )
                    and all(
                        isinstance(call.func, ast.Name)
                        and call.func.id in allowed_calls
                        for call in ast.walk(node)
                        if isinstance(call, ast.Call)
                    )
                ):
                    continue
            violations.append(
                f"{path.relative_to(ROOT.parent)}:{node.lineno}: {type(node).__name__}"
            )
    assert not violations, "Non-export code in package initializers:\n" + "\n".join(
        violations
    )


def test_every_package_has_readme():
    missing = [
        str(path.parent.relative_to(ROOT))
        for path in ROOT.rglob("__init__.py")
        if "assets" not in path.parts and not (path.parent / "README.md").is_file()
    ]
    assert not missing, missing


def test_package_readmes_do_not_use_template_filler():
    phrases = (
        "purpose: provides ",
        "surrounding mobilerun packages",
        "contents: package modules",
    )
    violations = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("README.md")
        if any(phrase in path.read_text().lower() for phrase in phrases)
    ]
    assert not violations, violations


def test_macro_utilities_expose_state_matching_and_logging():
    from mobilerun.macro.utils import (
        capture_fingerprint,
        compare_states,
        configure_macro_logging,
    )

    assert callable(capture_fingerprint)
    assert callable(compare_states)
    assert callable(configure_macro_logging)


def test_skills_utils_parse_frontmatter_description():
    from mobilerun.skills.utils import _description

    assert _description("---\ndescription: Example\n---\n") == "Example"


def test_macro_runtime_defaults_read_environment(monkeypatch):
    from mobilerun.macro import MacroPlayer

    monkeypatch.setenv("MOBILERUN_MACRO_STATE_TIMEOUT", "12")
    monkeypatch.setenv("MOBILERUN_MACRO_STATE_THRESHOLD", "0.4")
    monkeypatch.setenv("MOBILERUN_MACRO_DELAY", "2")
    monkeypatch.setenv("MOBILERUN_MACRO_POLL_INTERVAL", "0.3")
    player = MacroPlayer()
    assert (player.delay_between_actions, player.state_timeout) == (2, 12)
    assert (player.state_threshold, player.state_poll_interval) == (0.4, 0.3)
    monkeypatch.setenv("MOBILERUN_MACRO_STATE_TIMEOUT", "13")
    assert MacroPlayer().state_timeout == 13
    assert MacroPlayer(state_timeout=7).state_timeout == 7


def test_macro_matcher_threshold_reads_environment(monkeypatch):
    from mobilerun.macro.utils import compare_states
    from mobilerun.models import Fingerprint

    saved = Fingerprint(app="a", activity=None, elements=("id:x", "id:y"))
    current = Fingerprint(app="a", activity=None, elements=("id:x",))
    monkeypatch.setenv("MOBILERUN_MACRO_STATE_THRESHOLD", "0.51")
    assert not compare_states(saved, current)
    monkeypatch.setenv("MOBILERUN_MACRO_STATE_THRESHOLD", "0.5")
    assert compare_states(saved, current)


def test_macro_list_default_directory_reads_environment(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from mobilerun.macro.cli import macro

    path = tmp_path / "not-a-directory"
    path.write_text("file")
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(path))
    result = CliRunner().invoke(macro, ["list", "--json"])
    assert result.exit_code != 0
    assert "Not a directory" in result.output


def test_macro_replay_cli_defaults_read_environment(monkeypatch):
    import click

    from mobilerun.macro.cli.replay import replay

    monkeypatch.setenv("MOBILERUN_MACRO_DELAY", "3")
    monkeypatch.setenv("MOBILERUN_MACRO_STATE_TIMEOUT", "9")
    monkeypatch.setenv("MOBILERUN_MACRO_STATE_THRESHOLD", "0.7")
    options = {param.name: param for param in replay.params}
    with click.Context(replay) as context:
        assert options["delay"].get_default(context) == 3
        assert options["state_timeout"].get_default(context) == 9
        assert options["state_threshold"].get_default(context) == 0.7


def test_mcp_format_is_owned_by_mcp_and_registry_override(monkeypatch):
    from mobilerun.mcp.installer import _agent_format
    from mobilerun.mcp.settings import MCPFormat

    monkeypatch.setenv(
        "MOBILERUN_AGENTS",
        '{"custom":{"skills":null,"mcp":["x","y"],'
        '"mcp_format":{"servers_key":"custom_servers"}}}',
    )
    assert isinstance(_agent_format("custom"), MCPFormat)
    assert _agent_format("custom").servers_key == "custom_servers"


def test_sandbox_missing_dependency_has_install_hint():
    code = """
import importlib.abc
import sys
class BlockFunctions(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mobilerun_functions":
            raise ModuleNotFoundError("blocked", name=fullname)
sys.meta_path.insert(0, BlockFunctions())
import mobilerun
import mobilerun.cli.app
try:
    from mobilerun.sandbox import Context
except ImportError as exc:
    assert "pip install 'mobilerun[sandbox]'" in str(exc), str(exc)
else:
    raise AssertionError("sandbox import unexpectedly succeeded")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
