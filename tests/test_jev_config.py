import asyncio
import io
import shutil
from dataclasses import asdict, fields
from pathlib import Path

import pytest
import yaml
from rich.console import Console

import mobilerun.agent.jev.client as jev_client_module
import mobilerun.cli.doctor as doctor
from mobilerun.agent.jev.errors import (
    JevAuthError,
    JevConfigError,
    JevServiceError,
    JevTransportError,
)
from mobilerun.cli.doctor import CheckResult, Status, check_typesafe
from mobilerun.config_manager import ConfigLoader, env_keys
from mobilerun.config_manager.config_manager import JevConfig, MobileConfig

SECRET = "ts-test-secret-4242"
EXAMPLE_CONFIG = Path(__file__).parents[1] / "mobilerun" / "config_example.yaml"


@pytest.fixture(autouse=True)
def isolate_api_keys(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(env_keys, "AUTH_PROFILES_PATH", tmp_path / "auth-profiles.json")
    monkeypatch.setenv("TYPESAFE_API_KEY", "placeholder")
    monkeypatch.delenv("TYPESAFE_API_KEY")


@pytest.fixture
def doctor_output(monkeypatch) -> io.StringIO:
    buffer = io.StringIO()
    monkeypatch.setattr(doctor, "console", Console(file=buffer, width=200))
    return buffer


def _install_fake_client(monkeypatch, *, models=None, error=None) -> list:
    instances = []

    class FakeJevClient:
        def __init__(self, api_key, **kwargs):
            self.api_key = api_key
            self.kwargs = kwargs
            self.closed = False
            instances.append(self)

        async def list_models(self):
            if error is not None:
                raise error
            return list(models or ["jev-latest", "jev-preview"])

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr(jev_client_module, "JevClient", FakeJevClient, raising=False)
    return instances


def _jev_config(**jev) -> MobileConfig:
    return MobileConfig.from_dict({"agent": {"name": "jev", "jev": jev}})


# ── config ───────────────────────────────────────────────────────


def test_jev_config_round_trips_through_dicts() -> None:
    jev = {
        "model": "jev-1.14.0",
        "min_confidence": 0.5,
        "api_key_source": "env",
        "base_url": "https://typesafe.example.test",
        "timeout": 12.5,
        "max_retries": 1,
    }
    config = MobileConfig.from_dict({"agent": {"name": "jev", "jev": jev}})

    data = config.to_dict()

    assert config.agent.name == "jev"
    assert config.agent.jev == JevConfig(**jev)
    assert data["agent"]["jev"] == jev
    assert MobileConfig.from_dict(data).agent.jev == config.agent.jev


def test_jev_config_defaults_hold_no_key() -> None:
    config = MobileConfig.from_dict({"agent": {}})

    assert config.agent.jev == JevConfig()
    assert config.to_dict()["agent"]["jev"] == asdict(JevConfig())
    assert "api_key" not in {f.name for f in fields(JevConfig)}


def test_saved_config_keeps_jev_settings_without_key(monkeypatch, tmp_path) -> None:
    path = tmp_path / "config.yaml"
    monkeypatch.setattr(
        ConfigLoader, "get_user_config_path", classmethod(lambda cls: path)
    )
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    config = _jev_config(model="jev-preview", min_confidence=0.25)

    ConfigLoader.save(config)
    loaded = ConfigLoader.load(str(path))

    assert loaded.agent.name == "jev"
    assert loaded.agent.jev == config.agent.jev
    assert SECRET not in path.read_text()


def test_config_example_documents_jev(tmp_path) -> None:
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    copy = tmp_path / "config.yaml"
    shutil.copy(EXAMPLE_CONFIG, copy)

    config = ConfigLoader.load(str(copy))

    assert raw["agent"]["name"] == "mobilerun"
    assert set(raw["agent"]["jev"]) == {f.name for f in fields(JevConfig)}
    assert config.agent.name == "mobilerun"
    assert config.agent.jev == JevConfig()


def test_unknown_jev_keys_are_ignored() -> None:
    config = MobileConfig.from_dict(
        {"agent": {"name": "jev", "jev": {"model": "jev-preview", "future": 1}}}
    )

    assert config.agent.jev == JevConfig(model="jev-preview")


def test_empty_jev_section_uses_defaults() -> None:
    config = MobileConfig.from_dict({"agent": {"jev": None}})

    assert config.agent.jev == JevConfig()


def test_quoted_jev_numbers_in_yaml_are_converted(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "agent:\n"
        "  name: jev\n"
        "  jev:\n"
        '    min_confidence: "0.5"\n'
        '    timeout: "12"\n'
        '    max_retries: "1"\n'
    )

    jev = MobileConfig.from_yaml(str(path)).agent.jev

    assert (jev.min_confidence, jev.timeout, jev.max_retries) == (0.5, 12.0, 1)
    assert type(jev.min_confidence) is float
    assert type(jev.timeout) is float
    assert type(jev.max_retries) is int


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"min_confidence": 30}, "agent.jev.min_confidence must be between 0 and 1"),
        ({"min_confidence": -1}, "agent.jev.min_confidence must be between 0 and 1"),
        ({"min_confidence": "high"}, "agent.jev.min_confidence must be a number"),
        ({"min_confidence": "nan"}, "agent.jev.min_confidence must be a number"),
        ({"min_confidence": True}, "agent.jev.min_confidence must be a number"),
        ({"timeout": 0}, "agent.jev.timeout must be greater than 0"),
        ({"timeout": "30s"}, "agent.jev.timeout must be a number"),
        ({"timeout": "inf"}, "agent.jev.timeout must be a number"),
        ({"max_retries": -1}, "agent.jev.max_retries must be a whole number"),
        ({"max_retries": 1.5}, "agent.jev.max_retries must be a whole number"),
        ({"max_retries": None}, "agent.jev.max_retries must be a number"),
        ({"api_key_source": "shell"}, "agent.jev.api_key_source must be 'auto'"),
        ({"model": ""}, "agent.jev.model must be a non-empty string"),
        ({"model": None}, "agent.jev.model must be a non-empty string"),
        ({"base_url": " "}, "agent.jev.base_url must be a non-empty string"),
    ],
)
def test_invalid_jev_settings_are_rejected(settings, message) -> None:
    with pytest.raises(ValueError, match=message):
        _jev_config(**settings)
    with pytest.raises(ValueError, match=message):
        JevConfig(**settings)


def test_non_mapping_jev_section_is_rejected() -> None:
    with pytest.raises(ValueError, match="agent.jev must be a mapping"):
        MobileConfig.from_dict({"agent": {"jev": "jev-preview"}})


# ── doctor ───────────────────────────────────────────────────────


def test_check_typesafe_skips_when_jev_is_unused(monkeypatch) -> None:
    clients = _install_fake_client(monkeypatch)

    assert asyncio.run(check_typesafe(MobileConfig())) is None
    assert clients == []


def test_check_typesafe_fails_without_key_when_jev_selected(monkeypatch) -> None:
    clients = _install_fake_client(monkeypatch)

    result = asyncio.run(check_typesafe(_jev_config()))

    assert result.status is Status.FAIL
    assert result.message == "no API key"
    assert "mobilerun configure jev" in result.detail
    assert clients == []


def test_check_typesafe_passes_with_accepted_key(monkeypatch) -> None:
    clients = _install_fake_client(monkeypatch)
    env_keys.save_env_keys({"typesafe": SECRET})

    result = asyncio.run(check_typesafe(_jev_config()))

    assert result.status is Status.PASS
    assert result.message == "key accepted; models: jev-latest, jev-preview"
    assert clients[0].api_key == SECRET
    assert clients[0].closed


def test_check_typesafe_runs_when_key_exists_without_jev(monkeypatch) -> None:
    _install_fake_client(monkeypatch)
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)

    result = asyncio.run(check_typesafe(MobileConfig()))

    assert result.status is Status.PASS


def test_check_typesafe_honors_api_key_source(monkeypatch) -> None:
    clients = _install_fake_client(monkeypatch)
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)

    result = asyncio.run(check_typesafe(_jev_config(api_key_source="file")))

    assert result.status is Status.FAIL
    assert clients == []


def test_check_typesafe_fails_on_invalid_api_key_source(monkeypatch) -> None:
    clients = _install_fake_client(monkeypatch)
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    config = _jev_config()
    config.agent.jev.api_key_source = "shell"

    result = asyncio.run(check_typesafe(config))

    assert result.status is Status.FAIL
    assert result.detail == "agent.jev.api_key_source must be 'auto', 'env' or 'file'."
    assert clients == []


def test_check_typesafe_env_source_ignores_saved_key(monkeypatch) -> None:
    clients = _install_fake_client(monkeypatch)
    env_keys.save_env_keys({"typesafe": SECRET})
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    result = asyncio.run(check_typesafe(_jev_config(api_key_source="env")))

    assert result.status is Status.FAIL
    assert "agent.jev.api_key_source is 'env'" in result.detail
    assert "configure jev" not in result.detail
    assert clients == []


@pytest.mark.parametrize(
    ("model", "status"),
    [
        ("jev-preview", Status.PASS),
        ("jev-1.14.0", Status.PASS),
        ("jev-1.13", Status.WARN),
        ("gpt-5", Status.WARN),
    ],
)
def test_check_typesafe_flags_unlisted_models(monkeypatch, model, status) -> None:
    _install_fake_client(monkeypatch)
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)

    result = asyncio.run(check_typesafe(_jev_config(model=model)))

    assert result.status is status
    if status is Status.WARN:
        assert f"model {model} not listed" in result.message


def test_check_typesafe_fails_on_invalid_settings(monkeypatch) -> None:
    error = JevConfigError("TypeSafe base_url must use https for remote hosts.")
    _install_fake_client(monkeypatch, error=error)
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)

    result = asyncio.run(check_typesafe(_jev_config()))

    assert result.status is Status.FAIL
    assert result.message == "invalid settings"
    assert "base_url" in result.detail


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (JevAuthError("TypeSafe rejected the API key (HTTP 401).", status=401), "fail"),
        (JevTransportError("Could not reach TypeSafe (ConnectError)."), "warn"),
        (JevServiceError("TypeSafe failed (HTTP 529) after 1 attempts."), "warn"),
        (RuntimeError(f"unexpected {SECRET}"), "warn"),
    ],
)
def test_check_typesafe_errors_never_print_key(
    monkeypatch, doctor_output, error, status
) -> None:
    _install_fake_client(monkeypatch, error=error)
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)

    result = asyncio.run(check_typesafe(_jev_config()))
    doctor._print_result(result, debug=True)
    doctor._print_summary([result])

    assert result.status is Status(status)
    output = doctor_output.getvalue()
    assert "TypeSafe" in output
    assert SECRET not in output
    assert SECRET not in result.detail


def test_run_doctor_reports_typesafe_after_config(monkeypatch, doctor_output) -> None:
    _install_fake_client(monkeypatch)
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)

    async def sdk(debug):
        return CheckResult("SDK Version", Status.PASS, "ok")

    async def config_check(debug):
        return CheckResult("Config", Status.PASS, "ok"), _jev_config()

    async def adb_check(debug):
        return CheckResult("ADB", Status.FAIL, "adb not found in PATH")

    monkeypatch.setattr(doctor, "check_sdk_version", sdk)
    monkeypatch.setattr(doctor, "check_config", config_check)
    monkeypatch.setattr(doctor, "check_adb", adb_check)

    asyncio.run(doctor.run_doctor(debug=True))

    output = doctor_output.getvalue()
    assert output.index("Config") < output.index("TypeSafe") < output.index("ADB")
    assert "key accepted" in output
    assert SECRET not in output
