"""Library modules must not call the built-in print function."""

import ast
from pathlib import Path


def test_shared_library_has_no_browser_or_stdout_presentation():
    root = Path(__file__).resolve().parents[2] / "mobilerun"
    violations = []
    for directory in ("utils", "clients", "core"):
        for path in (root / directory).rglob("*.py"):
            # Opt-in checkout handlers are the only library presentation boundary.
            if path == root / "clients" / "cloud" / "checkout.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    modules = (
                        [alias.name for alias in node.names]
                        if isinstance(node, ast.Import)
                        else [node.module or ""]
                    )
                    if any(module.split(".")[0] == "webbrowser" for module in modules):
                        violations.append(
                            f"{path.relative_to(root)}:{node.lineno}: webbrowser"
                        )
                if isinstance(node, ast.Call) and ast.unparse(node.func) in {
                    "print",
                    "sys.stdout.write",
                }:
                    violations.append(f"{path.relative_to(root)}:{node.lineno}: stdout")
    assert not violations, violations


def test_library_does_not_call_print():
    root = Path(__file__).resolve().parents[2] / "mobilerun"
    violations = []
    for path in root.rglob("*.py"):
        # Opt-in checkout handlers deliberately print payment instructions.
        if path == root / "clients" / "cloud" / "checkout.py":
            continue
        if "cli" in path.relative_to(root).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not violations, violations
