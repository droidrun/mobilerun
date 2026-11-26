"""Config migration, asset resolution, and environment-backed storage paths."""

import ast
import inspect
import json
import os
import runpy
import stat
import sys
from copy import deepcopy
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import get_args

import pytest
import yaml
from click.testing import CliRunner
from pydantic import SecretStr

from mobilerun.cli.app import cli
from mobilerun.config.env import Secrets
from mobilerun.config.loader import ConfigLoader, OutdatedConfigError
from mobilerun.config.migrations import CURRENT_VERSION, migrate
from mobilerun.config.settings import Settings, get_settings
from mobilerun.macro.settings import MacroSettings
from mobilerun.mcp.settings import MCPSettings
from mobilerun.utils import paths
from mobilerun.utils.paths import PathResolver


def test_resolve_still_finds_bundled_assets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    asset = PathResolver.resolve(
        "assets/prompts/devices/expect.jinja2", must_exist=True
    )
    assert asset.is_file()


def test_legacy_prompt_handles_addon_without_spec(tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "mobilerun_agents", ModuleType("mobilerun_agents"))
    prompt = "config/prompts/missing.jinja2"
    assert PathResolver.resolve(prompt) == tmp_path / prompt
    with pytest.raises(FileNotFoundError):
        PathResolver.resolve(prompt, must_exist=True)


def test_legacy_prompt_propagates_other_finder_errors(monkeypatch):

    def broken_finder(name):
        raise ValueError("broken finder")

    monkeypatch.delitem(sys.modules, "mobilerun_agents", raising=False)
    monkeypatch.setattr(paths, "find_spec", broken_finder)
    with pytest.raises(ValueError, match="broken finder"):
        paths.PathResolver.resolve("config/prompts/missing.jinja2")


@pytest.mark.parametrize("bundle_location", ["base", "addon", "base_shadowed"])
def test_v8_preserves_bundled_prompt_resolution(tmp_path, monkeypatch, bundle_location):
    base = tmp_path / "base"
    addon = tmp_path / "mobilerun_agents"
    asset = "assets/prompts/split.jinja2"
    legacy = "config/prompts/split.jinja2"
    bundled = (addon if bundle_location == "addon" else base) / asset
    bundled.parent.mkdir(parents=True)
    bundled.touch()
    addon.mkdir(exist_ok=True)
    (addon / "__init__.py").write_text("raise AssertionError('unexpected import')")
    if bundle_location == "base_shadowed":
        shadow = tmp_path / asset
        shadow.parent.mkdir(parents=True)
        shadow.touch()
    monkeypatch.setattr(PathResolver, "get_project_root", lambda: base)
    monkeypatch.delitem(sys.modules, "mobilerun_agents", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    config = {
        "_version": 7,
        "agent": {"fast": {"system_prompt": legacy, "user_prompt": legacy}},
    }

    assert PathResolver.resolve(legacy, must_exist=True) == bundled
    migrated = migrate(config)["agent"]["fast"]
    for key in ("system_prompt", "user_prompt"):
        assert PathResolver.resolve(migrated[key], must_exist=True) == bundled
        assert migrated[key] == (asset if bundle_location == "base" else legacy)


@pytest.mark.parametrize("missing", ["mobilerun_agents", "other_dependency"])
def test_blocked_addon_is_treated_as_missing_only_for_its_own_name(
    monkeypatch, missing
):

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


def test_v8_leaves_missing_prompt_unchanged_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(PathResolver, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(paths, "find_spec", lambda name: None)
    config = {
        "_version": 7,
        "agent": {"fast_agent": {"user_prompt": "config/prompts/custom.jinja2"}},
    }
    result = migrate(config)
    assert (
        result["agent"]["fast_agent"]["user_prompt"] == "config/prompts/custom.jinja2"
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
    prompt = "config/prompts/system.jinja2"
    local = tmp_path / prompt
    local.parent.mkdir(parents=True)
    local.touch()
    package = tmp_path / "package"
    bundled = package / "assets/prompts/system.jinja2"
    bundled.parent.mkdir(parents=True)
    bundled.touch()
    monkeypatch.setattr(PathResolver, "get_project_root", lambda: package)
    config = {
        "_version": 7,
        "agent": {"fast_agent": {"system_prompt": prompt}},
    }
    assert migrate(config)["agent"]["fast_agent"]["system_prompt"] == prompt


def test_legacy_app_card_path_is_not_redirected(tmp_path, monkeypatch):
    addon = tmp_path / "mobilerun_agents"
    (addon / "assets/app_cards").mkdir(parents=True)
    (addon / "__init__.py").touch()
    monkeypatch.delitem(sys.modules, "mobilerun_agents", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    legacy_path = "config/app_cards"
    config = {
        "_version": 7,
        "agent": {"app_cards": {"app_cards_dir": legacy_path}},
    }

    assert PathResolver.resolve(legacy_path) == tmp_path / legacy_path
    assert migrate(config)["agent"]["app_cards"]["app_cards_dir"] == legacy_path


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


@pytest.mark.parametrize("dotenv", [False, True])
def test_macro_override_equal_to_default_has_env_source(monkeypatch, tmp_path, dotenv):

    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("MOBILERUN_MACRO_DIRECTORY", raising=False)
    directory = str(tmp_path / "macros")
    if dotenv:
        env_file = tmp_path / "settings.env"
        env_file.write_text(f"MOBILERUN_MACRO_DIRECTORY={directory}\n")
        monkeypatch.setitem(MacroSettings.model_config, "env_file", str(env_file))
    else:
        monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", directory)
    result = CliRunner().invoke(cli, ["macro", "path", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["source"] == "env"


@pytest.mark.parametrize(
    "group,variable,name",
    [
        ("macro", "MOBILERUN_MACRO_DIRECTORY", "macros"),
        ("credentials", "MOBILERUN_CREDENTIALS_FILE", "credentials.yaml"),
    ],
)
def test_empty_override_uses_default(monkeypatch, tmp_path, group, variable, name):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv(variable, "")
    result = CliRunner().invoke(cli, [group, "path", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "path": str(tmp_path / "config" / name),
        "exists": False,
        "source": "default",
    }
    if group == "macro":
        (tmp_path / "wrong").mkdir()
        (tmp_path / "wrong" / "macro.json").write_text("{}")
        result = CliRunner().invoke(cli, ["macro", "list", "--json"])
        assert json.loads(result.output) == []


def test_credentials_path_existing_directory_and_symlink(monkeypatch, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("MOBILERUN_CREDENTIALS_FILE", str(link))
    result = CliRunner().invoke(cli, ["credentials", "path", "--json"])
    assert json.loads(result.output) == {
        "path": str(link),
        "exists": True,
        "source": "env",
    }


@pytest.mark.parametrize(
    "group,variable",
    [
        ("macro", "MOBILERUN_MACRO_DIRECTORY"),
        ("credentials", "MOBILERUN_CREDENTIALS_FILE"),
    ],
)
def test_path_source_from_dotenv(monkeypatch, tmp_path, group, variable):

    monkeypatch.delenv(variable, raising=False)
    env_file = tmp_path / "settings.env"
    env_file.write_text(f"{variable}={tmp_path / 'custom'}\n")
    model = MacroSettings if group == "macro" else Settings
    monkeypatch.setitem(model.model_config, "env_file", str(env_file))
    result = CliRunner().invoke(cli, [group, "path", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["source"] == "env"


def test_macro_default_path_and_list(monkeypatch, tmp_path):
    monkeypatch.delenv("MOBILERUN_MACRO_DIRECTORY", raising=False)
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(cli, ["macro", "path", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "path": str(tmp_path / "macros"),
        "exists": False,
        "source": "default",
    }
    (tmp_path / "macros" / "demo").mkdir(parents=True)
    (tmp_path / "macros" / "demo" / "macro.json").write_text("{}")
    result = runner.invoke(cli, ["macro", "list", "--json"])
    assert result.exit_code == 0, result.output
    assert [entry["name"] for entry in json.loads(result.output)] == ["demo"]


def test_local_path_overrides_and_existence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", "relative")
    monkeypatch.setenv("MOBILERUN_CREDENTIALS_FILE", "creds.yaml")
    (tmp_path / "relative").mkdir()
    (tmp_path / "creds.yaml").touch()
    runner = CliRunner()
    for group, path in [
        ("macro", tmp_path / "relative"),
        ("credentials", tmp_path / "creds.yaml"),
    ]:
        result = runner.invoke(cli, [group, "path", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {
            "path": str(path),
            "exists": True,
            "source": "env",
        }
        assert runner.invoke(cli, [group, "path"]).output.strip() == f"{path} (exists)"


def test_credentials_default_json(monkeypatch, tmp_path):
    monkeypatch.delenv("MOBILERUN_CREDENTIALS_FILE", raising=False)
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))
    result = CliRunner().invoke(cli, ["credentials", "path", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "path": str(tmp_path / "credentials.yaml"),
        "exists": False,
        "source": "default",
    }


def test_macro_home_override_does_not_use_package_fallback(monkeypatch, tmp_path):

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", "~/missing")
    result = CliRunner().invoke(cli, ["macro", "path", "--json"])
    assert json.loads(result.output) == {
        "path": str(tmp_path / "missing"),
        "exists": False,
        "source": "env",
    }

    package = tmp_path / "package"
    (package / "relative" / "fallback").mkdir(parents=True)
    (package / "relative" / "fallback" / "macro.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", "relative")
    monkeypatch.setattr(PathResolver, "get_project_root", lambda: package)
    result = CliRunner().invoke(cli, ["macro", "list", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []


MODELS = (Settings, Secrets, MCPSettings, MacroSettings)


@pytest.mark.parametrize(
    "model,name",
    [(model, name) for model in MODELS for name in model.model_fields],
    ids=lambda value: value.__name__ if isinstance(value, type) else value,
)
def test_exact_documented_environment_field(monkeypatch, model, name):
    field = model.model_fields[name]
    assert name == name.upper()
    assert field.validation_alias is None
    assert field.description
    annotation = field.annotation
    types = get_args(annotation) or (annotation,)
    default = field.get_default(call_default_factory=True)
    if SecretStr in types:
        raw = "dummy-secret-for-test"
        expected = SecretStr(raw)
    elif Path in types:
        raw = "/dummy/settings-test"
        expected = Path(raw)
    elif isinstance(default, Enum):
        raw = default.value
        expected = default
    elif isinstance(default, bool):
        raw, expected = "false", False
    elif isinstance(default, int):
        raw, expected = "42", 42
    elif isinstance(default, float):
        raw, expected = "12.5", 12.5
    elif name == "MOBILERUN_AGENTS":
        from mobilerun.config.settings import AgentPaths

        raw = '{"test": {"skills": ["global", "project"], "mcp": null}}'
        expected = {"test": AgentPaths(skills=("global", "project"), mcp=None)}
    elif isinstance(default, dict):
        raw = '{"test": ["global", "project"]}'
        expected = {"test": ("global", "project")}
    elif isinstance(default, (list, tuple)):
        raw = json.dumps([0.1, 0.2])
        expected = type(default)([0.1, 0.2])
    else:
        raw = "debug" if name == "MOBILERUN_LOG_LEVEL" else "dummy-value"
        expected = raw
    monkeypatch.setenv(name, raw)
    assert getattr(model(), name) == expected


def test_base_models_do_not_own_agent_fields():
    assert set(Secrets.model_fields) == {"MOBILERUN_API_KEY", "MOBILERUN_DEVICE_TOKEN"}
    assert not (set(Settings.model_fields) & {"PHOENIX_URL", "PHOENIX_PROJECT_NAME"})


@pytest.mark.parametrize("model", MODELS)
def test_fields_are_single_line_with_attribute_docstrings(model):
    cls = ast.parse(inspect.getsource(model)).body[0]
    for index, node in enumerate(cls.body):
        if not isinstance(node, ast.AnnAssign):
            continue
        assert node.lineno == node.end_lineno, node.target.id
        description = cls.body[index + 1]
        assert isinstance(description, ast.Expr)
        assert isinstance(description.value, ast.Constant)
        assert isinstance(description.value.value, str)
        assert description.lineno == description.end_lineno
        assert description.lineno == node.end_lineno + 1
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "Field":
            assert {keyword.arg for keyword in node.value.keywords} <= {
                "gt",
                "ge",
                "le",
                "min_length",
                "default_factory",
            }
            assert node.value.keywords


def test_get_settings_reads_environment_on_every_call(monkeypatch):
    monkeypatch.setenv("MOBILERUN_CLOUD_API_TIMEOUT", "10")
    first = get_settings()
    monkeypatch.setenv("MOBILERUN_CLOUD_API_TIMEOUT", "20")
    assert first.MOBILERUN_CLOUD_API_TIMEOUT == 10
    assert get_settings().MOBILERUN_CLOUD_API_TIMEOUT == 20


def test_local_credentials_path_defaults_to_user_config_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("MOBILERUN_CREDENTIALS_FILE", raising=False)
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))
    assert get_settings().local_credentials_path == tmp_path / "credentials.yaml"


def test_local_credentials_path_expands_home_override(monkeypatch):
    monkeypatch.setenv("MOBILERUN_CREDENTIALS_FILE", "~/custom-credentials.yaml")
    assert (
        get_settings().local_credentials_path == Path.home() / "custom-credentials.yaml"
    )


def test_local_credentials_path_expands_default_directory(monkeypatch):
    monkeypatch.delenv("MOBILERUN_CREDENTIALS_FILE", raising=False)
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", "~/custom-config")
    assert (
        get_settings().local_credentials_path
        == Path.home() / "custom-config/credentials.yaml"
    )
