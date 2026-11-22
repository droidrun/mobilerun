"""CLI configuration attaches providers before agent/macro execution."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from mobilerun.cli.devices.base import selection
from mobilerun.core.devices.base import Device as DeviceMixin


@pytest.mark.parametrize("configured_path", [True, False])
def test_macro_cli_attaches_config_file_before_replay(
    tmp_path, monkeypatch, configured_path
):
    from mobilerun.macro.cli import macro

    module = importlib.import_module("mobilerun.macro.cli.replay")
    secrets = tmp_path / "credentials.yaml"
    secrets.write_text("login: private-value\n")
    secrets.chmod(0o600)
    path = tmp_path / "macro.json"
    path.write_text(
        json.dumps(
            {
                "macro_schema_version": "2.0",
                "actions": [{"action_type": "type_secret", "secret": "login"}],
            }
        )
    )
    config = SimpleNamespace(
        credentials={
            "enabled": True,
            **({"file_path": str(secrets)} if configured_path else {}),
        },
        device=SimpleNamespace(platform="android", use_tcp=False),
    )
    device = DeviceMixin()
    device.input_text = AsyncMock(return_value=True)
    device._input_text_raw = device.input_text
    device.state.refresh = AsyncMock(
        return_value=SimpleNamespace(
            phone_state={
                "focusedElement": {"className": "EditText"},
                "isEditable": True,
            }
        )
    )
    monkeypatch.setattr(module.ConfigLoader, "load", lambda _: config)
    monkeypatch.setattr(
        module, "get_settings", lambda: SimpleNamespace(local_credentials_path=secrets)
    )
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=device))
    result = CliRunner().invoke(
        macro, ["replay", str(path), "fake-serial", "--delay", "0"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    device.input_text.assert_awaited_once_with("private-value")
    assert "private-value" not in result.output
