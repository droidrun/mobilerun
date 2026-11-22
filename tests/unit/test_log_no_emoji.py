"""Library log messages should not contain decorative emoji."""

import ast
from pathlib import Path


def test_library_log_messages_have_no_emoji():
    root = Path(__file__).resolve().parents[2] / "mobilerun"
    violations = []
    for path in root.rglob("*.py"):
        if path == root / "logs" / "actions.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if node.func.attr not in {
                "debug",
                "info",
                "warning",
                "error",
                "exception",
                "critical",
            }:
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id not in {
                "logger",
                "log",
                "logging",
            }:
                continue
            for arg in node.args[:1]:
                for part in ast.walk(arg):
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        if any(
                            0x1F300 <= ord(c) <= 0x1FAFF
                            or 0x2300 <= ord(c) <= 0x23FF
                            or 0x2600 <= ord(c) <= 0x27BF
                            for c in part.value
                        ):
                            violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not violations, violations
