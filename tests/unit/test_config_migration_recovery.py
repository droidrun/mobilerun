"""Migration failures and cwd-specific prompt overrides."""

import runpy
import stat
import sys

import pytest
import yaml

from mobilerun.config.loader import ConfigLoader
from mobilerun.utils import paths
from mobilerun.utils.paths import PathResolver


def test_backup_symlink_stops_migration(tmp_path):
    path = tmp_path / "config.yaml"
    original = "_version: 7\ndevice:\n  serial: keep-me\n"
    path.write_text(original)
    backup = tmp_path / "config.yaml.v7.bak"
    backup.symlink_to(path)
    with pytest.raises(FileExistsError, match="regular file"):
        ConfigLoader.load(str(path))
    assert backup.is_symlink()
    assert path.read_text() == original


def test_partial_dump_preserves_config_and_retry(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    original = "_version: 7\ndevice:\n  serial: keep-me\n"
    path.write_text(original)
    initial_entries = set(tmp_path.iterdir())

    def fail_after_header(data, stream, **kwargs):
        stream.write("_version: 8\n")
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(yaml, "dump", fail_after_header)
        with pytest.raises(OSError, match="disk full"):
            ConfigLoader.load(str(path))

    backup = path.with_suffix(".yaml.v7.bak")
    assert path.read_text() == backup.read_text() == original
    assert set(tmp_path.iterdir()) == initial_entries | {backup}
    assert ConfigLoader.load(str(path)).device.serial == "keep-me"
    assert yaml.safe_load(path.read_text())["_version"] == 8
    assert backup.read_text() == original
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("content", ["_version: 8\n", "_version: 8\ndevice:\n"])
def test_minimal_current_config_loads_defaults(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content)
    assert ConfigLoader.load(str(path)).device.device_id == "auto"
    assert path.read_text() == content


def test_migrate_and_save_symlinked_config(tmp_path, monkeypatch):
    target = tmp_path / "dotfiles" / "mobile.yaml"
    target.parent.mkdir()
    original = "_version: 7\ndevice:\n  serial: keep-me\n"
    target.write_text(original)
    link = tmp_path / "config.yaml"
    link.symlink_to(target)
    config = ConfigLoader.load(str(link))
    assert link.is_symlink()
    assert yaml.safe_load(target.read_text())["_version"] == 8
    assert target.with_suffix(".yaml.v7.bak").read_text() == original
    monkeypatch.setattr(ConfigLoader, "get_user_config_path", lambda: link)
    config.device.serial = "changed"
    ConfigLoader.save(config)
    assert link.is_symlink()
    assert yaml.safe_load(target.read_text())["device"]["serial"] == "changed"


def test_standalone_resolver_reads_package_asset(tmp_path, monkeypatch):
    # Load the utility alone, with no registration from config.migrations.
    resolver = runpy.run_path(paths.__file__)["PathResolver"]
    monkeypatch.setattr(resolver, "get_project_root", lambda: tmp_path)
    bundled = tmp_path / "assets/prompts/test.jinja2"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("bundled")
    assert resolver.resolve("config/prompts/test.jinja2", must_exist=True) == bundled


def test_move_into_project_after_migration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt = "config/prompts/system.jinja2"
    addon = tmp_path / "split" / "mobilerun_agents"
    bundled = addon / "assets/prompts/system.jinja2"
    bundled.parent.mkdir(parents=True)
    (addon / "__init__.py").write_text("raise AssertionError('unexpected import')")
    bundled.write_text("bundled")
    monkeypatch.delitem(sys.modules, "mobilerun_agents", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path / "split"))
    project = tmp_path / "project"
    local = project / prompt
    local.parent.mkdir(parents=True)
    local.write_text("local")
    path = tmp_path / "user.yaml"
    path.write_text(f"_version: 7\nagent:\n  fast:\n    system_prompt: {prompt}\n")

    ConfigLoader.load(str(path))
    stored = yaml.safe_load(path.read_text())["agent"]["fast"]["system_prompt"]
    assert stored == prompt
    assert PathResolver.resolve(stored, must_exist=True) == bundled
    monkeypatch.chdir(project)
    assert PathResolver.resolve(stored, must_exist=True) == local
