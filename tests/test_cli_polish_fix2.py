"""Platform token, UI text and safe SDK-error contracts."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from click import ClickException
from click.testing import CliRunner
from mobilerun_core import UiState
from mobilerun_sdk import APIConnectionError, APIStatusError, Mobilerun

from mobilerun.cli import device_lifecycle, deviceauth, selection
from mobilerun.cli import run as run_module
from mobilerun.cli.main import cli
from mobilerun.config_manager import ConfigLoader, MobileConfig
from mobilerun.config_manager.config_manager import DeviceConfig


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr(ConfigLoader, "get_user_config_path", lambda: tmp_path / "config.yaml")
    for name in (
        "MOBILERUN_ANDROID_PORTAL_TOKEN", "MOBILERUN_IOS_PORTAL_TOKEN",
        "MOBILERUN_CONFIG", "MOBILERUN_DEVICE", "MOBILERUN_TARGET",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("platform,name,other_name", [
    ("android", "MOBILERUN_ANDROID_PORTAL_TOKEN", "MOBILERUN_IOS_PORTAL_TOKEN"),
    ("ios", "MOBILERUN_IOS_PORTAL_TOKEN", "MOBILERUN_ANDROID_PORTAL_TOKEN"),
])
def test_platform_tokens_override_config_without_crossing_platforms(
    monkeypatch, platform, name, other_name,
):
    other = "ios" if platform == "android" else "android"
    config = DeviceConfig(platform=platform, auth_token="config-token")
    assert config.resolve_auth_token() == "config-token"
    assert config.resolve_auth_token(platform) == "config-token"
    assert config.resolve_auth_token(other) is None
    selected = selection.Selection(
        MobileConfig(device=config), "http://localhost:9", other, f"local-{other}-http",
    )
    if other == "ios":
        assert selection.portal_token(selected) is None
    else:
        with pytest.raises(ClickException, match="MOBILERUN_ANDROID_PORTAL_TOKEN"):
            selection.portal_token(selected)
    monkeypatch.setenv(other_name, "other-platform-token")
    assert config.resolve_auth_token() == "config-token"
    monkeypatch.setenv(name, "selected-platform-token")
    assert config.resolve_auth_token() == "selected-platform-token"
    assert config.resolve_auth_token(other) == "other-platform-token"
    assert selection.portal_token(selected) == "other-platform-token"


@pytest.mark.parametrize("platform", ["", None, "unknown"])
def test_invalid_platform_has_no_token(platform):
    assert DeviceConfig(platform=platform, auth_token="config-token").resolve_auth_token() is None
    if platform is not None:
        assert DeviceConfig(auth_token="config-token").resolve_auth_token(platform) is None


@pytest.mark.parametrize("platform", ["android", "ios"])
def test_legacy_token_variable_is_ignored(monkeypatch, platform):
    # This removed variable must never act as a fallback for either platform.
    monkeypatch.setenv("MOBILERUN_DEVICE_TOKEN", "ignored-token")
    assert DeviceConfig(platform=platform).resolve_auth_token() is None
    assert DeviceConfig(platform=platform, auth_token="config-token").resolve_auth_token() == "config-token"


def test_env_masks_both_platform_tokens(monkeypatch):
    names = ("MOBILERUN_ANDROID_PORTAL_TOKEN", "MOBILERUN_IOS_PORTAL_TOKEN")
    for name in names:
        monkeypatch.setenv(name, "token-secret-marker")
    for args in (["env"], ["env", "-j"]):
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 0, result.output
        assert "token-secret-marker" not in result.output
        assert all(name in result.stdout for name in names)
        if "-j" in args:
            rows = {row["variable"]: row for row in json.loads(result.stdout)}
            assert all(rows[name]["value"] == "set" and rows[name]["source"] == "env"
                       for name in names)


@pytest.mark.parametrize("platform,name,other_name", [
    ("android", "MOBILERUN_ANDROID_PORTAL_TOKEN", "MOBILERUN_IOS_PORTAL_TOKEN"),
    ("ios", "MOBILERUN_IOS_PORTAL_TOKEN", "MOBILERUN_ANDROID_PORTAL_TOKEN"),
])
def test_env_uses_config_token_only_for_configured_platform(platform, name, other_name):
    config = MobileConfig(device=DeviceConfig(platform=platform, auth_token="config-secret-marker"))
    ConfigLoader.save(config)
    result = CliRunner().invoke(cli, ["env", "-j"])
    rows = {row["variable"]: row for row in json.loads(result.stdout)}
    assert rows[name]["value"] == "set"
    assert rows[name]["source"] == "config"
    assert rows[other_name]["value"] == "not set"
    assert rows[other_name]["source"] == "not set"
    assert "config-secret-marker" not in result.output


@pytest.mark.parametrize("ios_token", [None, "ios-env-token"])
def test_run_platform_override_does_not_reassign_android_config_token(monkeypatch, ios_token):
    config = MobileConfig(device=DeviceConfig(platform="android", auth_token="android-config-token"))
    ConfigLoader.save(config)
    if ios_token:
        monkeypatch.setenv("MOBILERUN_IOS_PORTAL_TOKEN", ios_token)

    class Handler:
        async def stream_events(self):
            if False:
                yield None

        def __await__(self):
            async def done():
                return SimpleNamespace(success=True, reason="done", steps=1)
            return done().__await__()

    agent = Mock(return_value=SimpleNamespace(run=lambda: Handler()))
    monkeypatch.setattr("mobilerun.MobileAgent", agent)
    monkeypatch.setattr(run_module, "_cleanup_android_keyboard", AsyncMock())
    result = CliRunner().invoke(cli, [
        "run", "task", "-P", "ios", "-d", "http://localhost:8080", "-j",
    ])
    assert result.exit_code == 0, result.output
    resolved = agent.call_args.kwargs["config"].device
    assert resolved.platform == "ios"
    assert resolved.resolve_auth_token() == ios_token
    assert resolved.resolve_auth_token("android") is None


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
EXPECTED_ALL_LINES = [
    "0\tFrameLayout\t\t\t0,0,100,200\t",
    "1\tButton\tSave this now\tapp:id/save\t1,2,30,40\tclickable,focused",
    "2\tView\t\t\t\t",
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
    expected = EXPECTED_ALL_LINES if all_elements else EXPECTED_LINES
    assert result.stdout.splitlines() == expected
    assert all(line.count("\t") == 5 for line in result.stdout.splitlines())
    assert result.stderr == "App: Test App; screen: 100x200\n"
    device.ui_json.assert_not_called()


def test_ui_json_remains_raw_core_state(monkeypatch):
    import mobilerun_core

    device = Mock()
    device.ui_json.return_value = json.dumps(UI_STATE)
    monkeypatch.setattr(mobilerun_core.Mobilerun, "connect", lambda *a, **kw: device)
    result = CliRunner().invoke(cli, ["device", "ui", "-d", "serial", "-j"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == UI_STATE
    assert result.stderr == ""
    device.ui_json.assert_called_once_with()
    device.ui.assert_not_called()


def test_ui_uses_description_for_whitespace_and_sanitizes_every_control(monkeypatch):
    import mobilerun_core

    controls = "".join(chr(code) for code in (*range(32), *range(127, 160))) + "\u2028\u2029\u200e"
    state = UiState({"a11y_tree": {"children": [
        {"className": "Button", "text": " \t\n ", "contentDescription": "Description"},
        {"className": "TextView", "text": "before" + controls + "after",
         "resourceId": "res\x1b[31m", "bounds": [1, 2, 3, 4]},
    ]}})
    device = Mock()
    device.ui.return_value = state
    monkeypatch.setattr(mobilerun_core.Mobilerun, "connect", lambda *a, **kw: device)
    result = CliRunner().invoke(cli, ["device", "ui", "-d", "serial"])
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "1\tButton\tDescription\t\t\t",
        "2\tTextView\tbefore" + " " * len(controls) + "after\tres [31m\t\t",
    ]


@pytest.mark.parametrize("state", [
    UiState({"a11y_tree": None, "ime_tree": [{"text": "keyboard"}]}),
    SimpleNamespace(dead_tree=True),
    SimpleNamespace(a11y_tree_available=False),
])
def test_ui_unavailable_is_a_failure(monkeypatch, state):
    import mobilerun_core

    device = Mock()
    device.ui.return_value = state
    monkeypatch.setattr(mobilerun_core.Mobilerun, "connect", lambda *a, **kw: device)
    result = CliRunner().invoke(cli, ["device", "ui", "-d", "serial"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: The device UI tree is unavailable.\n"


def test_ui_all_and_json_is_a_usage_error_before_connecting(monkeypatch):
    import mobilerun_core

    connect = Mock()
    monkeypatch.setattr(mobilerun_core.Mobilerun, "connect", connect)
    result = CliRunner().invoke(cli, ["device", "ui", "-d", "serial", "--all", "-j"])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "--all cannot be combined with --json" in result.stderr
    connect.assert_not_called()


REQUEST_KEY = "private-key-marker"
ENV_KEY = "env-key-marker"
SDK_PATH = "/v1/devices/0python-false-1000"


@pytest.fixture
def sdk_error_cases():
    cases = []
    for kind, status, body, message, code, hint in [
        ("status", 401, {"detail": "Not authorized", "title": "Unauthorized", "code": "unauthorized"},
         "Not authorized", "unauthorized", "mobilerun auth login"),
        ("status", 403, {"title": "Forbidden"}, "Forbidden", None, "MOBILERUN_CLOUD_API_KEY"),
        ("status", 404, {"detail": "Device not found"},
         "Device not found", None, "mobilerun device list -t cloud"),
        ("status", 422, {"errors": [{"message": "Invalid device id", "code": "validation_error"}]},
         "Invalid device id", "validation_error", None),
        ("status", 500, {}, "Internal Server Error", None, None),
        ("status", 422, {
            "detail": f"Device 10 python false {REQUEST_KEY} {ENV_KEY}", "code": "0_python_false",
        }, "Device 10 python false [redacted] [redacted]", "0_python_false", None),
        ("connection", None, None, "Could not connect to the API", None, "https://api.example.test"),
        ("timeout", None, None, "Request timed out", None, "https://api.example.test"),
    ]:
        def respond(request, kind=kind, status=status, body=body):
            if kind == "connection":
                raise httpx.ConnectError(f"raw connection {REQUEST_KEY}", request=request)
            if kind == "timeout":
                raise httpx.ReadTimeout(f"raw timeout {REQUEST_KEY}", request=request)
            return httpx.Response(status, json={
                **body, "raw_secret": REQUEST_KEY, "request_headers": dict(request.headers),
            })

        with Mobilerun(
            api_key=REQUEST_KEY, base_url="https://api.example.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)), max_retries=0,
        ) as client:
            try:
                client.devices.retrieve("0python-false-1000", extra_query={"token": REQUEST_KEY})
            except (APIStatusError, APIConnectionError) as error:
                assert error.request.headers["x-stainless-lang"] == "python"
                assert error.request.headers["x-stainless-async"] == "false"
                assert error.request.headers["x-stainless-retry-count"] == "0"
                cases.append((error, message, status, code, hint))
            else:
                pytest.fail("The mock SDK request must fail")
    return cases


def invoke_api_error(monkeypatch, error, *flags):
    monkeypatch.setenv("MOBILERUN_CLOUD_API_KEY", ENV_KEY)
    monkeypatch.setattr(deviceauth, "load_credentials", lambda: None)
    monkeypatch.setattr(device_lifecycle, "core_client", lambda *a, **kw: SimpleNamespace(
        list_devices=Mock(side_effect=error), close=Mock(),
    ))
    result = CliRunner().invoke(cli, ["device", "list", "-t", "cloud", "-v", *flags])
    assert result.exit_code == 1, result.output
    for forbidden in (
        REQUEST_KEY, ENV_KEY, "Authorization", "request_headers", "raw_secret", "Traceback",
        "raw connection", "raw timeout", "core wrapper", "?token",
    ):
        assert forbidden not in result.output
    assert result.stderr.startswith(f"GET {SDK_PATH}\nError: ")
    return result


def test_api_errors_text(monkeypatch, sdk_error_cases):
    for error, message, status, _, hint in sdk_error_cases:
        result = invoke_api_error(monkeypatch, error)
        expected = f"GET {SDK_PATH}\nError: {message}"
        if status is not None:
            expected += f" (HTTP {status})"
        assert result.stderr.startswith(expected + "\n")
        if hint:
            assert hint in result.stderr
        assert result.stdout == ""


def test_api_errors_json(monkeypatch, sdk_error_cases):
    for error, message, status, code, _ in sdk_error_cases:
        result = invoke_api_error(monkeypatch, error, "-j")
        expected = {"error": message, "status": status, "code": code}
        if status in {401, 403}:
            expected["credential"] = "env"
        assert json.loads(result.stdout) == expected


def test_wrapped_api_error_uses_sdk_cause(monkeypatch, sdk_error_cases):
    error, message, status, code, _ = sdk_error_cases[0]
    wrapped = RuntimeError(f"core wrapper {REQUEST_KEY}")
    wrapped.__cause__ = error
    result = invoke_api_error(monkeypatch, wrapped, "-j")
    assert json.loads(result.stdout) == {"error": message, "status": status, "code": code, "credential": "env"}
