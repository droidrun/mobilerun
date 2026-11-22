"""Every environment-backed field is its own documented, exact environment name."""

import ast
import inspect
import json
from enum import Enum
from pathlib import Path
from typing import get_args

import pytest
from pydantic import SecretStr

from mobilerun.config.env import Secrets
from mobilerun.config.settings import Settings, get_settings
from mobilerun.macro.settings import MacroSettings
from mobilerun.mcp.settings import MCPSettings

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
        raw, expected = "4321", 4321
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
