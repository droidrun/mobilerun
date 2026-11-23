import os
from pathlib import Path

import pytest
import yaml

from mobilerun.config.loader import ConfigLoader, OutdatedConfigError
from mobilerun.config.migrations import CURRENT_VERSION


def test_example_omits_obsolete_bundled_app_card_directory():
    example = (
        Path(__file__).resolve().parents[2] / "assets/examples/config_example.yaml"
    )
    data = yaml.safe_load(example.read_text())
    assert "app_cards" not in data.get("agent", {})


def test_existing_app_card_settings_are_preserved(tmp_path):
    path = tmp_path / "config.yaml"
    cards = {"enabled": True, "app_cards_dir": "my-cards"}
    path.write_text(
        yaml.safe_dump({"_version": CURRENT_VERSION, "agent": {"app_cards": cards}})
    )

    config = ConfigLoader.load(str(path))

    assert config.to_dict()["agent"]["app_cards"] == cards
    assert yaml.safe_load(path.read_text())["agent"]["app_cards"] == cards


def test_unversioned_config_migrates_and_preserves_original(tmp_path):
    path = tmp_path / "config.yaml"
    original = "# old config\ndevice:\n  serial: my-device\nagent:\n  codeact:\n    vision: true\n"
    path.write_text(original)

    config = ConfigLoader.load(str(path))

    assert config.device.serial == "my-device"
    data = yaml.safe_load(path.read_text())
    assert data["_version"] == CURRENT_VERSION
    assert data["agent"]["fast_agent"]["vision"] is True
    backup = tmp_path / "config.yaml.v1.bak"
    assert backup.read_text() == original
    if os.name != "nt":
        assert backup.stat().st_mode & 0o777 == 0o600


def test_current_config_is_not_rewritten_or_backed_up(tmp_path):
    path = tmp_path / "config.yaml"
    original = (
        f"# keep formatting\n_version: {CURRENT_VERSION}\ndevice:\n  serial: current\n"
    )
    path.write_text(original)
    assert ConfigLoader.load(str(path)).device.serial == "current"
    assert path.read_text() == original
    assert not list(tmp_path.glob("config.yaml.v*.bak"))


def test_existing_backup_is_not_overwritten(tmp_path):
    path = tmp_path / "config.yaml"
    original = "device:\n  serial: new\n"
    path.write_text(original)
    backup = tmp_path / "config.yaml.v1.bak"
    backup.write_text("previous backup")

    with pytest.raises(FileExistsError):
        ConfigLoader.load(str(path))

    assert path.read_text() == original
    assert backup.read_text() == "previous backup"


@pytest.mark.parametrize("content", ["[broken", "- not a mapping", "_version: nope"])
def test_invalid_config_raises_outdated_error_without_rewriting(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content)
    with pytest.raises(OutdatedConfigError):
        ConfigLoader.load(str(path))
    assert path.read_text() == content
