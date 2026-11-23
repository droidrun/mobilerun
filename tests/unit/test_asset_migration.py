"""Migration and resolution of bundled runtime assets."""

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from mobilerun.config.migrations import migrate


def test_v8_finds_split_addon_assets_and_preserves_absent(tmp_path):
    base = Path(__file__).resolve().parents[2]
    addon = tmp_path / "addon" / "mobilerun" / "agents"
    addon.mkdir(parents=True)
    (addon / "__init__.py").write_text("")
    (addon / "assets" / "prompts").mkdir(parents=True)
    (addon / "assets" / "prompts" / "split.jinja2").write_text("split")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(base), str(addon.parents[1])))
    script = """
import mobilerun, sys
mobilerun.__path__ = [p for p in mobilerun.__path__ if not p.endswith('/mobilerun/mobilerun')]
if sys.argv[1] == 'absent': mobilerun.__path__ = []
from mobilerun.config.migrations.v008_bundled_assets import migrate
config = {'agent': {'fast': {'system_prompt': 'config/prompts/split.jinja2'}}}
actual = migrate(config)['agent']['fast']['system_prompt']
assert actual == ('config/prompts/split.jinja2' if sys.argv[1] == 'absent' else 'assets/prompts/split.jinja2'), actual
"""
    # Import migration before pruning the base package path.
    script = script.replace(
        "import mobilerun, sys",
        "import mobilerun, sys\nimport mobilerun.config.migrations.v008_bundled_assets",
    )
    for mode in ("present", "absent"):
        result = subprocess.run(
            [sys.executable, "-c", script, mode],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("addon_present", [True, False])
def test_v8_migrates_only_bundled_paths_and_is_idempotent(
    tmp_path, monkeypatch, addon_present
):
    from mobilerun.config.migrations import v008_bundled_assets

    monkeypatch.chdir(tmp_path)
    addon = tmp_path / "addon"
    if addon_present:
        (addon / "assets/prompts/fast_agent").mkdir(parents=True)
        (addon / "assets/prompts/fast_agent/system.jinja2").write_text("prompt")
        (addon / "assets/app_cards").mkdir(parents=True)
    from types import SimpleNamespace

    monkeypatch.setattr(
        v008_bundled_assets,
        "find_spec",
        lambda name: (
            SimpleNamespace(submodule_search_locations=[str(addon)])
            if addon_present
            else None
        ),
    )
    config = {
        "_version": 7,
        "agent": {
            "fast_agent": {
                "system_prompt": "config/prompts/fast_agent/system.jinja2",
                "user_prompt": "config/prompts/custom.jinja2",
            },
            "app_cards": {"app_cards_dir": "config/app_cards"},
        },
    }
    result = migrate(config)
    assert (
        result["agent"]["fast_agent"]["system_prompt"]
        == ("assets" if addon_present else "config")
        + "/prompts/fast_agent/system.jinja2"
    )
    assert (
        result["agent"]["fast_agent"]["user_prompt"] == "config/prompts/custom.jinja2"
    )
    assert result["agent"]["app_cards"]["app_cards_dir"] == (
        "assets/app_cards" if addon_present else "config/app_cards"
    )
    assert migrate(deepcopy(result)) == result


def test_v6_chain_keeps_manager_auto_selection():
    result = migrate(
        {
            "_version": 6,
            "agent": {
                "manager": {"system_prompt": "config/prompts/manager/system.jinja2"}
            },
        }
    )
    assert result["agent"]["manager"]["system_prompt"] is None
    assert result["_version"] == 8


def test_migration_preserves_cwd_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "config/app_cards"
    local.mkdir(parents=True)
    config = {
        "_version": 7,
        "agent": {"app_cards": {"app_cards_dir": "config/app_cards"}},
    }
    assert migrate(config)["agent"]["app_cards"]["app_cards_dir"] == "config/app_cards"
