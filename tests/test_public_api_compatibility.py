import tomllib
from pathlib import Path

import pytest

import mobilerun
from mobilerun.agent.droid import MobileAgentState

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _read_project(path: Path) -> dict:
    with path.open("rb") as project_file:
        return tomllib.load(project_file)["project"]


def test_mobile_agent_state_is_exported_from_package_root():
    assert mobilerun.MobileAgentState is MobileAgentState
    assert "MobileAgentState" in mobilerun.__all__


def test_legacy_droid_agent_state_alias_resolves_to_canonical_class():
    with pytest.warns(
        DeprecationWarning,
        match="DroidAgentState has been renamed to MobileAgentState",
    ):
        assert mobilerun.DroidAgentState is MobileAgentState


def test_legacy_droid_agent_state_can_be_imported_from_package_root():
    with pytest.warns(
        DeprecationWarning,
        match="DroidAgentState has been renamed to MobileAgentState",
    ):
        from mobilerun import DroidAgentState

    assert DroidAgentState is MobileAgentState


def test_compatibility_package_uses_matching_exact_pins():
    mobilerun_project = _read_project(REPOSITORY_ROOT / "pyproject.toml")
    droidrun_project = _read_project(REPOSITORY_ROOT / "compat" / "pyproject.toml")
    version = mobilerun_project["version"]

    assert droidrun_project["version"] == version
    assert droidrun_project["dependencies"] == [f"mobilerun=={version}"]

    mobilerun_extras = mobilerun_project["optional-dependencies"]
    droidrun_extras = droidrun_project["optional-dependencies"]
    assert mobilerun_extras["deepseek"] == []
    for extra in ("anthropic", "deepseek", "langfuse", "typesafe", "dev"):
        assert droidrun_extras[extra] == [f"mobilerun[{extra}]=={version}"]
