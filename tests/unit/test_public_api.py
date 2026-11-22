import tomllib
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _read_project(path: Path) -> dict:
    with path.open("rb") as project_file:
        return tomllib.load(project_file)["project"]


def test_base_resolves_legacy_macro_player_with_deprecation():
    import mobilerun
    from mobilerun.macro import MacroPlayer

    assert "MacroPlayer" not in mobilerun.__all__
    with pytest.warns(DeprecationWarning, match="mobilerun.MacroPlayer"):
        assert mobilerun.MacroPlayer is MacroPlayer


def test_removed_base_config_legacy_name():
    from mobilerun.config import user_config

    assert not hasattr(user_config, "DroidConfig")


def test_compatibility_package_uses_matching_exact_pins():
    mobilerun_project = _read_project(REPOSITORY_ROOT / "pyproject.toml")
    droidrun_project = _read_project(REPOSITORY_ROOT / "compat" / "pyproject.toml")
    version = mobilerun_project["version"]

    assert droidrun_project["version"] == version
    requirements = {
        req.name: req for req in map(Requirement, droidrun_project["dependencies"])
    }
    assert len(droidrun_project["dependencies"]) == 2
    assert set(requirements) == {"mobilerun", "mobilerun-agents"}
    assert str(requirements["mobilerun"].specifier) == f"=={version}"
    agents_specifier = list(requirements["mobilerun-agents"].specifier)
    assert len(agents_specifier) == 1 and agents_specifier[0].operator == "=="
    assert "*" not in agents_specifier[0].version
    assert all(not req.extras and req.marker is None for req in requirements.values())

    mobilerun_extras = mobilerun_project["optional-dependencies"]
    droidrun_extras = droidrun_project["optional-dependencies"]
    assert set(droidrun_extras) == {"anthropic", "deepseek", "langfuse", "dev"}
    for extra, dependencies in droidrun_extras.items():
        assert len(dependencies) == 1
        requirement = Requirement(dependencies[0])
        package = "mobilerun" if extra == "dev" else "mobilerun-agents"
        assert requirement.name == package
        assert requirement.extras == {extra}
        assert requirement.marker is None
        assert requirement.specifier == requirements[package].specifier
        if package == "mobilerun":
            assert extra in mobilerun_extras


def test_compatibility_package_agent_pins_match_installed_metadata():
    try:
        agents_metadata = metadata.metadata("mobilerun-agents")
    except metadata.PackageNotFoundError:
        pytest.skip("mobilerun-agents is not installed; cannot validate its metadata")

    droidrun_project = _read_project(REPOSITORY_ROOT / "compat" / "pyproject.toml")
    requirements = [Requirement(dep) for dep in droidrun_project["dependencies"]]
    agents_requirement = next(
        req for req in requirements if req.name == "mobilerun-agents"
    )
    assert str(agents_requirement.specifier) == f"=={agents_metadata['Version']}"
    agents_extras = set(agents_metadata.get_all("Provides-Extra") or [])
    for extra, dependencies in droidrun_project["optional-dependencies"].items():
        requirement = Requirement(dependencies[0])
        if requirement.name == "mobilerun-agents":
            assert extra in agents_extras


def test_library_import_does_not_install_logging_handler():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import mobilerun, logging; assert not logging.getLogger('mobilerun').handlers",
        ],
        check=True,
    )
