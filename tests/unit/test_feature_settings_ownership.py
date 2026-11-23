"""Feature settings and optional dependencies stay out of the base package."""

import os
import subprocess
import sys


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
class BlockRoutines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mobilerun_routines":
            raise ModuleNotFoundError("blocked", name=fullname)
sys.meta_path.insert(0, BlockRoutines())
import mobilerun
import mobilerun.cli.app
try:
    import mobilerun.sandbox
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
