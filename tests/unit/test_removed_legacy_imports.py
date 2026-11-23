"""Removed import compatibility must not silently return."""

import importlib

import pytest


@pytest.mark.parametrize(
    "name",
    ["droidrun", "mobilerun.agent", "mobilerun.config_manager", "mobilerun.tools"],
)
def test_removed_imports(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_unknown_root_export():
    import mobilerun

    with pytest.raises(AttributeError):
        _ = mobilerun.definitely_not_a_public_export


def test_removed_root_alias():
    import mobilerun

    with pytest.raises(AttributeError):
        _ = mobilerun.MacroPlayer
