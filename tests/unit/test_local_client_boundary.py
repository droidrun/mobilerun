"""Local client helpers stay function-only and independent of core/cloud."""

import ast
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[2] / "mobilerun" / "clients" / "local"


def test_local_client_layout_and_import_boundaries():
    expected = {
        "credentials.py",
        "apk.py",
        "codes.py",
        "emulator.py",
        "login_store.py",
    }
    assert expected <= {path.name for path in LOCAL.glob("*.py")}
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
