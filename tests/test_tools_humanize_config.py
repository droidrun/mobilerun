import pytest

from mobilerun.config_manager.config_manager import MobileConfig


def test_tools_humanize_is_read_from_config():
    config = MobileConfig.from_dict({"tools": {"humanize": True}})

    assert config.tools.humanize is True


def test_tools_humanize_defaults_to_false():
    assert MobileConfig.from_dict({}).tools.humanize is False


def test_tools_stealth_fails_with_rename_message():
    with pytest.raises(
        ValueError, match="`tools.stealth` was renamed to `tools.humanize`"
    ):
        MobileConfig.from_dict({"tools": {"stealth": True}})
