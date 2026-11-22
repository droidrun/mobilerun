"""The shared framework must be usable without installing the agents package."""

import ast
import subprocess
import sys
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "mobilerun"
EXCEPTIONS = {
    "_legacy/aliases.py",  # Lazy deprecated root exports; runtime isolation is checked below.
    "_legacy/config.py",  # Optional legacy typed-config facade.
    "_legacy/config_manager/__init__.py",  # Optional legacy config facade.
    "cli/app.py",  # CLI root registers the agent command groups until agents becomes its own package (DRO-3973).
}


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
                violations.append(f"{path.relative_to(ROOT)}:{line}")
    assert not violations, violations


def test_clients_and_utils_do_not_import_core():
    for folder in ("clients", "utils"):
        for path in (ROOT / folder).rglob("*.py"):
            for line, target in _references(path):
                assert not target.startswith("mobilerun.core"), f"{path}:{line}"


def test_utils_import_only_utils_from_mobilerun():
    """Standalone utilities must not depend on domain packages."""
    violations = []
    for path in (ROOT / "utils").rglob("*.py"):
        for line, target in _references(path):
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
message = "mobilerun.agents"
import mobilerun.agents.operator
from mobilerun.agents import AgentOperator
import_module("mobilerun.agents.cli")
find_spec("mobilerun.agents")
__import__("mobilerun.agents.tools")
LEGACY_MODULES = {"old": "mobilerun.agents.legacy"}
LEGACY_MODULES.update({"other": "mobilerun.agents.extra"})
""")
    targets = [target for _, target in _agent_import_references(path)]
    assert targets.count("mobilerun.agents") == 2
    assert "mobilerun.agents.operator" in targets
    assert "mobilerun.agents.cli" in targets
    assert "mobilerun.agents.tools" in targets
    assert "mobilerun.agents.legacy" in targets
    assert "mobilerun.agents.extra" in targets


def test_base_does_not_import_agents():
    violations = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if relative.parts[0] == "agents" or str(relative) in EXCEPTIONS:
            continue
        for line, target in _agent_import_references(path):
            if (
                relative.as_posix() == "addons/agents.py"
                and target == "mobilerun.agents"
                and 'find_spec("mobilerun.agents")'
                in path.read_text().splitlines()[line - 1]
            ):
                continue  # Discovery is intentional and does not import the add-on.
            if "mobilerun.agents" in target:
                violations.append(f"{relative}:{line}")
    assert not violations, violations


def test_shared_packages_do_not_import_features():
    violations = []
    for folder in ("core", "clients", "config", "utils", "errors", "logs"):
        for path in (ROOT / folder).rglob("*.py"):
            for line, target in _references(path):
                if any(
                    "mobilerun." + feature in target
                    for feature in ("macro", "sandbox", "skills", "agents")
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{line}: {target}")
    assert not violations, violations


def test_independent_features_do_not_import_each_other_or_agents():
    violations = []
    for feature in ("macro", "sandbox", "skills"):
        forbidden = {"macro", "sandbox", "skills", "agents"} - {feature}
        for path in (ROOT / feature).rglob("*.py"):
            for line, target in _references(path):
                if any("mobilerun." + other in target for other in forbidden):
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
    for feature in ("macro", "agents", "skills", "sandbox"):
        for path in (ROOT / feature).rglob("*.py"):
            for line, target in _references(path):
                if target == "mobilerun.cli" or target.startswith("mobilerun.cli."):
                    assert any(
                        target == base or target.startswith(base + ".")
                        for base in ("mobilerun.cli.base", "mobilerun.cli.devices.base")
                    ), f"{path}:{line}: {target}"


def test_import_macro_is_agent_free():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import mobilerun.macro, sys; assert not any(m.startswith('mobilerun.agents') for m in sys.modules)",
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


def test_import_mobilerun_is_agent_free():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import mobilerun, sys; assert not any(m.startswith('mobilerun.agents') for m in sys.modules)",
        ],
        check=True,
    )


def test_compat_resolver_does_not_eagerly_import_agents():
    """Only the explicit deprecated export seam may reference agent targets."""
    tree = ast.parse((ROOT / "_legacy/aliases.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            assert all(not name.startswith("mobilerun.agents") for name in names)


def test_root_cli_help_does_not_load_agents():
    script = """
from click.testing import CliRunner
from mobilerun.addons.agents import agents_available
from mobilerun.cli.app import cli
import sys
result = CliRunner().invoke(cli, ["--help"])
assert result.exit_code == 0, result.output
names = {"account", "device", "doctor", "macro", "skills"}
if agents_available():
    names.add("agent")
rows = [line.strip().split(maxsplit=1) for line in result.output.splitlines()
        if line.startswith("  ") and line.strip().split()[0] in names | {"agent"}]
assert {row[0] for row in rows} == names, rows
assert all(len(row) == 2 and row[1] for row in rows), rows
assert not any(m.startswith("mobilerun.agents") for m in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_macro_help_and_agent_dry_run_do_not_load_agents(tmp_path):
    macro = tmp_path / "macro.json"
    macro.write_text('{"macro_schema_version": "2.0", "actions": []}')
    script = """
import sys
from click.testing import CliRunner
from mobilerun.cli.app import cli
runner = CliRunner()
for args in (["macro", "--help"], ["macro", "replay", "--help"],
             ["macro", "replay", sys.argv[1], "--on-mismatch", "agent", "--dry-run"]):
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert not any(name.startswith("mobilerun.agents") for name in sys.modules)
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
    support = ROOT.parent / "tests" / "support"
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
