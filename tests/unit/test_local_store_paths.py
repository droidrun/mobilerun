import json

import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli


@pytest.mark.parametrize("dotenv", [False, True])
def test_macro_override_equal_to_default_has_env_source(monkeypatch, tmp_path, dotenv):
    from mobilerun.macro.settings import MacroSettings

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
    from mobilerun.config.settings import Settings
    from mobilerun.macro.settings import MacroSettings

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
    from mobilerun.utils.paths import PathResolver

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
