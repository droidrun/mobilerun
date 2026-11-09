import asyncio

import pytest

from mobilerun.cli import doctor
from mobilerun.config_manager.loader import ConfigLoader

CONFIG = "_version: {version}\nagent:\n  max_steps: 7\n"


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    path = tmp_path / "user"
    monkeypatch.setattr(
        ConfigLoader, "get_user_config_dir", classmethod(lambda cls: path)
    )
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)
    monkeypatch.delenv("DROIDRUN_CONFIG", raising=False)
    return path


@pytest.fixture
def env_config(tmp_path):
    from mobilerun.config_manager.migrations import CURRENT_VERSION

    path = tmp_path / "env.yaml"
    path.write_text(CONFIG.format(version=CURRENT_VERSION), encoding="utf-8")
    return path


def test_missing_env_config_fails_without_writing_defaults(
    user_dir, tmp_path, monkeypatch
):
    missing = tmp_path / "missing.yaml"
    monkeypatch.setenv("MOBILERUN_CONFIG", str(missing))

    with pytest.raises(FileNotFoundError, match="MOBILERUN_CONFIG points to"):
        ConfigLoader.load()

    assert not (user_dir / "config.yaml").exists()


def test_missing_deprecated_env_config_names_the_variable(
    user_dir, tmp_path, monkeypatch
):
    monkeypatch.setenv("DROIDRUN_CONFIG", str(tmp_path / "missing.yaml"))

    with pytest.warns(DeprecationWarning):
        with pytest.raises(FileNotFoundError, match="DROIDRUN_CONFIG points to"):
            ConfigLoader.load()


def test_existing_env_config_is_loaded(user_dir, env_config, monkeypatch):
    monkeypatch.setenv("MOBILERUN_CONFIG", str(env_config))

    assert ConfigLoader.load().agent.max_steps == 7
    assert not (user_dir / "config.yaml").exists()


def test_unset_env_config_still_creates_user_config(user_dir):
    ConfigLoader.load()

    assert (user_dir / "config.yaml").exists()


def test_doctor_reports_missing_env_config(user_dir, tmp_path, monkeypatch):
    missing = tmp_path / "missing.yaml"
    monkeypatch.setenv("MOBILERUN_CONFIG", str(missing))

    result, config = asyncio.run(doctor.check_config(debug=False))

    assert result.status == doctor.Status.FAIL
    assert "MOBILERUN_CONFIG points to" in result.message
    assert config is None
    assert not (user_dir / "config.yaml").exists()


def test_doctor_reports_env_config_path(user_dir, env_config, monkeypatch):
    monkeypatch.setenv("MOBILERUN_CONFIG", str(env_config))

    result, config = asyncio.run(doctor.check_config(debug=False))

    assert result.status == doctor.Status.PASS
    assert result.message == str(env_config)
    assert config.agent.max_steps == 7
