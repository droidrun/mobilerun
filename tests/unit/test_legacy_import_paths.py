"""The 0.7 SDK paths remain lazy aliases, not duplicate class definitions."""

import subprocess
import sys
from pathlib import Path

import pytest


def run_legacy(code):
    compat = Path(__file__).resolve().parents[2] / "compat"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(compat)!r})\n{code}",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("root", ["droidrun", "mobilerun"])
def test_old_modules_preserve_identity_and_warn_once(root):
    run_legacy(f"""
import importlib, warnings
import {root}
paths = {{
    "config_manager": "mobilerun._legacy.config_manager",
    "config_manager.config_manager": "mobilerun._legacy.config",
    "tools": "mobilerun._legacy.tools",
    "tools.driver": "mobilerun_core_local.driver",
    "tools.ui": "mobilerun.core.state",
    "tools.ui.state": "mobilerun.core.state.ui_state",
    "tools.filters.concise_filter": "mobilerun.core.state.filters.concise_filter",
    "tools.formatters.indexed_formatter": "mobilerun.core.state.formatters.indexed_formatter",
    "tools.helpers.coordinate": "mobilerun.utils.coordinates",
    "log_handlers": "mobilerun.logs.handlers",
    "macro.replay": "mobilerun.macro.player",
}}
for old, new in paths.items():
    name = "{root}." + old
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            legacy = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            assert old.startswith("config_manager") and "pip install mobilerun-agents" in str(exc), (old, exc)
            continue
        assert legacy is importlib.import_module(new), name
        assert legacy.__spec__.name == new, (name, legacy.__spec__)
        importlib.import_module(name)
    notices = [w for w in caught if issubclass(w.category, DeprecationWarning)
               and str(w.message).startswith(name + " is deprecated;")]
    assert len(notices) == 1, (name, notices)
""")


def test_every_compat_mapping_has_a_real_destination():
    run_legacy("""
import importlib
import mobilerun
from mobilerun._legacy.aliases import LEGACY_MODULES
for old, new in LEGACY_MODULES.items():
    try:
        legacy = importlib.import_module("mobilerun." + old)
    except ModuleNotFoundError as exc:
        assert "pip install mobilerun-agents" in str(exc), (old, exc)
    else:
        assert legacy is importlib.import_module(new), old
""")


def test_plain_imports_do_not_load_optional_agent_modules():
    run_legacy("""
import droidrun
assert "mobilerun" not in sys.modules
import mobilerun
assert "mobilerun.agents" not in sys.modules
assert not any(name.startswith("llama_index") for name in sys.modules)
""")
