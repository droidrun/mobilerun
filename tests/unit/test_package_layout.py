"""Package initializers contain exports, not implementation."""

import ast
from pathlib import Path

# Selected initializers expose lazy public names without importing optional dependencies.
INITIALIZER_EXCEPTIONS = {
    "__init__.py": "Lazy public exports",
    "core/__init__.py": "Lazy public core exports",
    "core/devices/__init__.py": "Lazy public device exports",
    "sandbox/__init__.py": "Actionable error when the optional routines dependency is missing",
}


def test_initializers_only_export_names():
    root = Path(__file__).resolve().parents[2] / "mobilerun"
    violations = []
    for path in sorted(root.rglob("__init__.py")):
        # CLI subgroup packages intentionally define their Click group at the
        # package entry point; command implementations live in sibling modules.
        if path.parent.relative_to(root).as_posix() in {
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
            relative = path.relative_to(root).as_posix()
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
                f"{path.relative_to(root.parent)}:{node.lineno}: {type(node).__name__}"
            )
    assert not violations, "Non-export code in package initializers:\n" + "\n".join(
        violations
    )


def test_every_package_has_readme():
    root = Path(__file__).resolve().parents[2] / "mobilerun"
    missing = [
        str(path.parent.relative_to(root))
        for path in root.rglob("__init__.py")
        if "assets" not in path.parts and not (path.parent / "README.md").is_file()
    ]
    assert not missing, missing


def test_package_readmes_do_not_use_template_filler():
    root = Path(__file__).resolve().parents[2] / "mobilerun"
    phrases = (
        "purpose: provides ",
        "surrounding mobilerun packages",
        "contents: package modules",
    )
    violations = [
        str(path.relative_to(root))
        for path in root.rglob("README.md")
        if any(phrase in path.read_text().lower() for phrase in phrases)
    ]
    assert not violations, violations
