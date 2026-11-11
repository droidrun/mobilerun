"""Platform token, UI text and safe SDK-error contracts."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from click.testing import CliRunner
from mobilerun_core import UiState
from mobilerun_sdk import APIConnectionError, APIStatusError, APITimeoutError

from mobilerun.cli import device_lifecycle, selection
from mobilerun.cli.main import cli
from mobilerun.config_manager import ConfigLoader, MobileConfig
from mobilerun.config_manager.config_manager import PORTAL_TOKEN_ENV_VARS, DeviceConfig


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr(ConfigLoader, "get_user_config_path", lambda: tmp_path / "config.yaml")
    for name in (*PORTAL_TOKEN_ENV_VARS.values(), "MOBILERUN_CONFIG", "MOBILERUN_DEVICE", "MOBILERUN_TARGET"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("platform", ["android", "ios"])
def test_platform_tokens_override_config_without_crossing_platforms(monkeypatch, platform):
    other = "ios" if platform == "android" else "android"
    config = DeviceConfig(platform=platform, auth_token="config-token")
    monkeypatch.setenv(PORTAL_TOKEN_ENV_VARS[other], "other-platform-token")
    assert config.resolve_auth_token() == "config-token"
    monkeypatch.setenv(PORTAL_TOKEN_ENV_VARS[platform], "selected-platform-token")
    assert config.resolve_auth_token() == "selected-platform-token"
    assert config.resolve_auth_token(other) == "other-platform-token"
    selected = selection.Selection(MobileConfig(device=config), "http://localhost:9", other, f"local-{other}-http")
    assert selection.portal_token(selected) == "other-platform-token"


@pytest.mark.parametrize("platform", ["android", "ios"])
def test_legacy_token_variable_is_ignored(monkeypatch, platform):
    monkeypatch.setenv("MOBILERUN_" + "DEVICE_TOKEN", "ignored-token")
    assert DeviceConfig(platform=platform).resolve_auth_token() is None
    assert DeviceConfig(platform=platform, auth_token="config-token").resolve_auth_token() == "config-token"


def test_env_masks_both_platform_tokens(monkeypatch):
    for name in PORTAL_TOKEN_ENV_VARS.values():
        monkeypatch.setenv(name, "token-secret-marker")
    for args in (["env"], ["env", "-j"]):
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 0, result.output
        assert "token-secret-marker" not in result.output
        assert all(name in result.stdout for name in PORTAL_TOKEN_ENV_VARS.values())
        if "-j" in args:
            rows = {row["variable"]: row for row in json.loads(result.stdout)}
            assert all(rows[name]["value"] == "set" and rows[name]["source"] == "env" for name in PORTAL_TOKEN_ENV_VARS.values())


def test_env_uses_config_token_only_for_configured_platform():
    config = MobileConfig(device=DeviceConfig(platform="ios", auth_token="config-secret-marker"))
    ConfigLoader.save(config)
    result = CliRunner().invoke(cli, ["env", "-j"])
    rows = {row["variable"]: row for row in json.loads(result.stdout)}
    assert rows[PORTAL_TOKEN_ENV_VARS["ios"]]["value"] == "set"
    assert rows[PORTAL_TOKEN_ENV_VARS["ios"]]["source"] == "config"
    assert rows[PORTAL_TOKEN_ENV_VARS["android"]]["value"] == "not set"
    assert "config-secret-marker" not in result.output


UI_STATE = {
    "a11y_tree": {
        "className": "android.widget.FrameLayout",
        "boundsInScreen": {"left": 0, "top": 0, "right": 100, "bottom": 200},
        "children": [
            {"className": "android.widget.Button", "text": "Save\tthis\nnow", "resourceId": "app:id/save",
             "isClickable": True, "isFocused": True,
             "boundsInScreen": {"left": 1, "top": 2, "right": 30, "bottom": 40}},
            {"className": "android.view.View", "children": [
                {"class_name": "android.widget.CheckBox", "content_description": "Accept\rterms",
                 "is_checkable": True, "checked": True, "selected": True, "enabled": False,
                 "bounds": {"x": 4, "y": 5, "width": 20, "height": 10}},
                {"className": "android.widget.EditText", "resourceId": "app:id/input", "editable": True},
            ]},
            {"className": "android.widget.ScrollView", "isScrollable": True},
        ],
    },
    "phone_state": {"currentApp": "Test\nApp"},
    "device_context": {"screen_bounds": {"width": 100, "height": 200}},
}
EXPECTED_LINES = [
    "1\tButton\tSave this now\tapp:id/save\t1,2,30,40\tclickable,focused",
    "3\tCheckBox\tAccept terms\t\t4,5,24,15\tcheckable,checked,selected,disabled",
    "4\tEditText\t\tapp:id/input\t\teditable",
    "5\tScrollView\t\t\t\tscrollable",
]


@pytest.mark.parametrize("device_id", ["serial", "123e4567-e89b-12d3-a456-426614174000"])
@pytest.mark.parametrize("all_elements", [False, True])
def test_ui_text_is_one_line_per_core_node(monkeypatch, device_id, all_elements):
    import mobilerun_core

    device = Mock()
    device.ui.return_value = UiState(UI_STATE)
    monkeypatch.setattr(mobilerun_core.Mobilerun, "connect", lambda *a, **kw: device)
    monkeypatch.setenv("MOBILERUN_CLOUD_API_KEY", "fake-key")
    args = ["device", "ui", "-d", device_id] + (["--all"] if all_elements else [])
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    expected = EXPECTED_LINES[:]
    if all_elements:
        expected.insert(0, "0\tFrameLayout\t\t\t0,0,100,200\t")
        expected.insert(2, "2\tView\t\t\t\t")
    assert result.stdout.splitlines() == expected
    assert all(line.count("\t") == 5 for line in result.stdout.splitlines())
    assert result.stderr == "App: Test App; screen: 100x200\n"
    device.ui.assert_called_once_with(filter=False)
    device.ui_json.assert_not_called()


def test_ui_json_remains_raw_core_state(monkeypatch):
    import mobilerun_core

    device = Mock()
    device.ui_json.return_value = json.dumps(UI_STATE)
    monkeypatch.setattr(mobilerun_core.Mobilerun, "connect", lambda *a, **kw: device)
    result = CliRunner().invoke(cli, ["device", "ui", "-d", "serial", "-j", "--all"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == UI_STATE
    assert result.stderr == ""
    device.ui_json.assert_called_once_with()
    device.ui.assert_not_called()


@pytest.mark.parametrize("kind,status,body,message,code,hint", [
    ("status", 401, {"detail": "Not authorized", "title": "Unauthorized", "code": "unauthorized"}, "Not authorized", "unauthorized", "mobilerun auth login"),
    ("status", 403, {"title": "Forbidden"}, "Forbidden", None, "MOBILERUN_CLOUD_API_KEY"),
    ("status", 404, {"detail": "Device not found"}, "Device not found", None, "mobilerun device list -t cloud"),
    ("status", 422, {"errors": [{"message": "Invalid device id", "code": "validation_error"}]}, "Invalid device id", "validation_error", None),
    ("connection", None, None, "Could not connect to the API", None, "https://api.example.test/v1"),
    ("timeout", None, None, "Request timed out", None, "https://api.example.test/v1"),
])
@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_api_errors_are_safe_and_consistent(monkeypatch, kind, status, body, message, code, hint, json_output, wrapped):
    marker = "private-key-marker"
    header_marker = "private-header-marker"
    request = httpx.Request("GET", f"https://api.example.test/v1/devices/missing?token={marker}",
                            headers={"Authorization": f"Bearer {marker}", "X-Private": header_marker})
    if status is not None:
        body = {**body, "raw_secret": marker, "request_headers": dict(request.headers)}
        error = APIStatusError(f"raw body {body!r}", response=httpx.Response(status, request=request), body=body)
    elif kind == "timeout":
        error = APITimeoutError(request)
    else:
        error = APIConnectionError(message=f"raw connection {marker} {header_marker}", request=request)
    def fail(**kwargs):
        if wrapped:
            raise RuntimeError(f"core wrapper {marker}") from error
        raise error
    monkeypatch.setattr(device_lifecycle, "core_client", lambda *a, **kw: SimpleNamespace(list_devices=fail))
    monkeypatch.setenv("MOBILERUN_CLOUD_API_KEY", marker)
    flags = ["-j"] if json_output else []
    result = CliRunner().invoke(cli, ["device", "list", "-t", "cloud", "--api-url", "https://api.example.test/v1", "-v", *flags])
    assert result.exit_code == 1, result.output
    assert marker not in result.output and header_marker not in result.output
    assert "Authorization" not in result.output and "request_headers" not in result.output
    assert "raw body" not in result.output and "core wrapper" not in result.output
    assert "Traceback" not in result.output and "?token" not in result.output
    assert "GET /v1/devices/missing\n" in result.stderr
    assert f"Error: {message}" + (f" (HTTP {status})" if status else "") in result.stderr
    if hint:
        assert hint in result.stderr
    if json_output:
        assert json.loads(result.stdout) == {"error": message, "status": status, "code": code}
    else:
        assert result.stdout == ""


def test_api_error_message_redacts_echoed_key(monkeypatch):
    marker = "echoed-private-key"
    request = httpx.Request("GET", "https://api.example.test/v1/devices", headers={"Authorization": f"Bearer {marker}"})
    body = {"detail": f"Invalid credential {marker}"}
    error = APIStatusError("do not print raw body", response=httpx.Response(401, request=request), body=body)
    monkeypatch.setattr(device_lifecycle, "core_client", lambda *a, **kw: SimpleNamespace(list_devices=Mock(side_effect=error)))
    result = CliRunner().invoke(cli, ["device", "list", "-t", "cloud", "-j"])
    assert result.exit_code == 1
    assert marker not in result.output
    assert json.loads(result.stdout)["error"] == "Invalid credential [redacted]"
