from __future__ import annotations

import asyncio


def run(coro):
    return asyncio.run(coro)


def test_auth_token_defaults_to_none(monkeypatch):
    from mobilerun.config_manager.config_manager import DeviceConfig

    monkeypatch.delenv("MOBILERUN_IOS_PORTAL_TOKEN", raising=False)
    assert DeviceConfig(platform="ios").resolve_auth_token() is None


def test_auth_token_reads_config_value(monkeypatch):
    from mobilerun.config_manager.config_manager import DeviceConfig

    monkeypatch.delenv("MOBILERUN_IOS_PORTAL_TOKEN", raising=False)
    assert DeviceConfig(platform="ios", auth_token="from-config").resolve_auth_token() == "from-config"


def test_auth_token_env_overrides_config(monkeypatch):
    from mobilerun.config_manager.config_manager import DeviceConfig

    monkeypatch.setenv("MOBILERUN_IOS_PORTAL_TOKEN", "from-env")
    assert DeviceConfig(platform="ios", auth_token="from-config").resolve_auth_token() == "from-env"


LOCAL_PORTAL_STATE = {
    "a11y_tree": {
        "boundsInScreen": {"left": 0, "top": 0, "right": 393, "bottom": 852},
        "className": "Application",
        "resourceId": "",
        "text": "",
        "contentDescription": "",
        "packageName": "com.apple.Preferences",
        "children": [
            {
                "boundsInScreen": {"left": 16, "top": 180, "right": 377, "bottom": 224},
                "className": "Button",
                "resourceId": "Privacy",
                "text": "Privacy & Security",
                "contentDescription": "",
                "packageName": "com.apple.Preferences",
                "children": [],
            }
        ],
    },
    "phone_state": {
        "currentApp": "Settings",
        "packageName": "com.apple.Preferences",
        "keyboardVisible": False,
    },
    "device_context": {
        "screen_bounds": {"width": 393, "height": 852},
        "filtering_params": {"min_element_size": 5, "overlay_offset": 0},
        "display_metrics": {"widthPixels": 393, "heightPixels": 852},
    },
}


class FakeLocalPortalDriver:
    """Duck-typed stand-in for IOSPortalHttpDriver in provider tests."""

    platform = "iOS"

    async def get_ui_tree(self):
        return LOCAL_PORTAL_STATE


def test_android_state_provider_consumes_local_portal_state():
    from mobilerun.tools.filters import DetailedFilter
    from mobilerun.tools.formatters import IndexedFormatter
    from mobilerun.tools.ui.provider import AndroidStateProvider

    provider = AndroidStateProvider(
        FakeLocalPortalDriver(),
        tree_filter=DetailedFilter(),
        tree_formatter=IndexedFormatter(),
    )
    state = run(provider.get_state())

    assert state.screen_width == 393
    assert state.screen_height == 852
    assert state.elements, "expected elements parsed from the portal a11y tree"
    assert "Privacy & Security" in state.formatted_text


def test_android_state_provider_vision_contract_uses_point_space():
    from mobilerun.tools.filters import ConciseFilter
    from mobilerun.tools.formatters import IndexedFormatter
    from mobilerun.tools.ui.provider import AndroidStateProvider

    provider = AndroidStateProvider(
        FakeLocalPortalDriver(),
        tree_filter=ConciseFilter(),
        tree_formatter=IndexedFormatter(),
        vision_enabled=True,
    )
    state = run(provider.get_state())

    # 393x852 points fit under the legacy 2048 max-side cap, so the model
    # space equals the point space and taps need no rescaling.
    assert state.coordinate_contract_active
    assert (state.model_screenshot_width, state.model_screenshot_height) == (393, 852)
    assert (state.coordinate_scale_x, state.coordinate_scale_y) == (1.0, 1.0)


def test_device_commands_ui_returns_core_state(monkeypatch):
    import json

    from click.testing import CliRunner
    from mobilerun_core import Mobilerun

    from mobilerun.cli.main import cli
    class Device:
        def ui_json(self):
            return json.dumps(LOCAL_PORTAL_STATE)
    monkeypatch.setattr(Mobilerun, "connect", lambda *a, **kw: Device())
    result = CliRunner().invoke(cli, ["device", "ui", "-d", "http://127.0.0.1:8080", "-P", "ios", "-j"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == LOCAL_PORTAL_STATE


def test_device_commands_route_ios_through_facade(tmp_path, monkeypatch):
    from mobilerun_core import Mobilerun

    from mobilerun.cli import device_commands

    created = {}
    def connect(self, device_id, **kwargs):
        created.update(kwargs)
        return object()
    monkeypatch.setattr(Mobilerun, "connect", connect)
    monkeypatch.setenv("MOBILERUN_IOS_PORTAL_TOKEN", "sekret")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("_version: 7\ndevice:\n  platform: ios\n")
    with device_commands.device_driver({"device": "http://127.0.0.1:8080", "config_path": str(cfg),
                                        "platform": "ios", "api_url": "https://example.test/v1"}):
        pass
    assert created["url"] == "http://127.0.0.1:8080"
    assert created["token"] == "sekret"
    assert created["backend"] == "local-ios-http"


def test_device_commands_discover_when_no_device(tmp_path, monkeypatch):
    from mobilerun_core import Mobilerun

    from mobilerun.cli import device_commands

    created = {}
    async def discover():
        return "http://127.0.0.1:8081"
    def connect(self, device_id, **kwargs):
        created.update(kwargs)
        return object()
    monkeypatch.setattr("mobilerun_core_local.driver.ios.discover_ios_device", discover)
    monkeypatch.setattr(Mobilerun, "connect", connect)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("_version: 7\ndevice:\n  platform: ios\n")
    with device_commands.device_driver({"config_path": str(cfg), "platform": "ios", "api_url": "https://example.test/v1"}):
        pass
    assert created["url"] == "http://127.0.0.1:8081"


def test_android_state_provider_normalized_keeps_point_tap_bounds():
    from mobilerun.tools.filters import DetailedFilter
    from mobilerun.tools.formatters import IndexedFormatter
    from mobilerun.tools.ui.provider import AndroidStateProvider

    provider = AndroidStateProvider(
        FakeLocalPortalDriver(),
        tree_filter=DetailedFilter(),
        tree_formatter=IndexedFormatter(),
        use_normalized=True,
    )
    state = run(provider.get_state())

    button = next(e for e in state.elements if e["text"] == "Privacy & Security")
    assert button["bounds"] == "16,180,377,224"
    assert button["displayBounds"] == "40,211,959,262"
    assert "(40,211,959,262)" in state.formatted_text
    assert state.get_element_coords(button["index"]) == (196, 202)
