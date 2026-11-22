"""The task boundary uses the SDK's declared exception contract."""

import ast
from pathlib import Path


def test_task_boundary_does_not_probe_sdk_error_body():
    path = Path(__file__).resolve().parents[2] / "mobilerun/clients/cloud/tasks.py"
    tree = ast.parse(path.read_text())
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "body"
        for node in ast.walk(tree)
    )
