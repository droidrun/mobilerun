"""Migration and resolution of bundled runtime assets."""

import sys
from copy import deepcopy

import pytest

from mobilerun.config.migrations import migrate


def test_legacy_prompt_handles_addon_without_spec(tmp_path, monkeypatch):
    from types import ModuleType

    from mobilerun.utils.paths import PathResolver

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "mobilerun_agents", ModuleType("mobilerun_agents"))
    prompt = "config/prompts/missing.jinja2"
    assert PathResolver.resolve(prompt) == tmp_path / prompt
    with pytest.raises(FileNotFoundError):
        PathResolver.resolve(prompt, must_exist=True)


def test_legacy_prompt_propagates_other_finder_errors(monkeypatch):
    from mobilerun.utils import paths

    def broken_finder(name):
        raise ValueError("broken finder")

    monkeypatch.delitem(sys.modules, "mobilerun_agents", raising=False)
    monkeypatch.setattr(paths, "find_spec", broken_finder)
    with pytest.raises(ValueError, match="broken finder"):
        paths.PathResolver.resolve("config/prompts/missing.jinja2")


def test_v8_finds_split_addon_assets_without_importing(tmp_path, monkeypatch):
    addon = tmp_path / "mobilerun_agents"
    bundled = addon / "assets/prompts/split.jinja2"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("split")
    (addon / "__init__.py").write_text("raise AssertionError('unexpected import')")
    monkeypatch.delitem(sys.modules, "mobilerun_agents", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    config = {
        "_version": 7,
        "agent": {"fast": {"system_prompt": "config/prompts/split.jinja2"}},
    }
    assert (
        migrate(config)["agent"]["fast"]["system_prompt"]
        == "assets/prompts/split.jinja2"
    )


@pytest.mark.parametrize("missing", ["mobilerun_agents", "other_dependency"])
def test_blocked_addon_is_treated_as_missing_only_for_its_own_name(
    monkeypatch, missing
):
    from pathlib import Path

    from mobilerun.utils import paths

    def blocked(name):
        assert name == "mobilerun_agents"
        raise ModuleNotFoundError(name=missing)

    monkeypatch.setattr(paths, "find_spec", blocked)
    prompt = Path("config/prompts/blocked-test.jinja2")
    if missing != "mobilerun_agents":
        with pytest.raises(ModuleNotFoundError):
            paths.PathResolver.bundled_asset(prompt)
    else:
        assert paths.PathResolver.bundled_asset(prompt) is None


@pytest.mark.parametrize("addon_present", [True, False])
def test_v8_migrates_only_bundled_paths_and_is_idempotent(
    tmp_path, monkeypatch, addon_present
):
    from mobilerun.utils import paths

    monkeypatch.chdir(tmp_path)
    addon = tmp_path / "addon"
    if addon_present:
        (addon / "assets/prompts/fast_agent").mkdir(parents=True)
        (addon / "assets/prompts/fast_agent/system.jinja2").write_text("prompt")
        (addon / "assets/app_cards").mkdir(parents=True)
    from types import SimpleNamespace

    monkeypatch.setattr(
        paths,
        "find_spec",
        lambda name: (
            SimpleNamespace(submodule_search_locations=[str(addon)])
            if name == "mobilerun_agents" and addon_present
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
