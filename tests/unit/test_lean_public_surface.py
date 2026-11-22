"""Approved breaking API: public actions, private framework machinery."""

import inspect

from mobilerun.core.devices import Device

DEVICE_NAMES = {
    "supported_operations",
    "extensions",
    "actions",
    "context",
    "cloud_id",
    "unbind_extension",
    "close",
    "connect",
    "delete_file",
    "do",
    "drag",
    "expect",
    "get_clipboard",
    "get_location",
    "get_language",
    "get_timezone",
    "identifier",
    "input_text",
    "install_apk",
    "install_app",
    "list_apps",
    "list_files",
    "mailbox",
    "number",
    "open_url",
    "platform",
    "press_button",
    "press_key_code",
    "pull_file",
    "push_file",
    "reset_location",
    "screenshot",
    "set_clipboard",
    "set_language",
    "set_location",
    "set_timezone",
    "start_app",
    "state",
    "stop_app",
    "supported_buttons",
    "swipe",
    "tap",
    "tap_element",
    "uninstall_app",
    "wait",
}


def test_exact_device_cloud_and_pool_surfaces():
    from mobilerun import CloudDevice, DevicePool

    assert {n for n in dir(Device) if not n.startswith("_")} == DEVICE_NAMES
    assert {n for n in dir(CloudDevice) if not n.startswith("_")} == DEVICE_NAMES | {
        "info",
        "name",
        "reboot",
        "rename",
        "reset",
        "resume",
        "stop",
        "terminate",
        "wait_ready",
    }
    pool = DevicePool([Device()])
    assert {n for n in dir(pool) if not n.startswith("_")} == (
        DEVICE_NAMES - {"identifier", "pull_file", "cloud_id"}
    ) | {"devices", "bind_each"}


def test_exact_root_and_core_exports():
    import mobilerun
    import mobilerun.core
    import mobilerun.core.devices

    devices = {
        "Device",
        "AndroidDevice",
        "CloudDevice",
        "IOSDevice",
        "RemoteDevice",
        "DevicePool",
        "RecordingDevice",
        "StealthDevice",
    }
    proxies = {
        "CloudRotatingProxy",
        "CloudFixedProxy",
        "CloudSocks5Proxy",
        "LocalProxy",
    }
    assert set(mobilerun.core.devices.__all__) == devices
    assert set(mobilerun.core.__all__) == devices | proxies | {"Workspace"}
    assert set(mobilerun.__all__) == devices | proxies | {
        "Workspace",
        "__version__",
        "DeviceConfig",
        "LoggingConfig",
        "MobileConfig",
        "TelemetryConfig",
        "TracingConfig",
    }
    for name in mobilerun.__all__:
        assert getattr(mobilerun, name) is not None
    for name in proxies:
        assert getattr(mobilerun, name) is getattr(mobilerun.core, name)
        assert getattr(mobilerun, name) is getattr(mobilerun.core.proxy, name)
    for prefix in ("Rotating", "Fixed", "Socks5", "Http"):
        name = prefix + "Proxy"
        assert not hasattr(mobilerun, name)
        assert not hasattr(mobilerun.core, name)
        assert not hasattr(mobilerun.core.proxy, name)
    for name in {
        "AndroidDriver",
        "DeviceDriver",
        "RecordingDriver",
        "VisualRemoteDevice",
    }:
        assert name not in vars(mobilerun)


def test_device_has_no_public_plumbing():
    removed = {
        "attach_state_provider",
        "attach_state_provider_factory",
        "default_state_provider",
        "use_default_state_provider",
        "attach_operator",
        "default_operator",
        "operator",
        "unwrap",
        "initialize",
        "ensure_connected",
        "needs_initialization",
        "get_apps",
        "list_packages",
        "ui_state",
        "recording",
        "state_recovery",
        "input_coordinate_size",
        "get_ui_tree",
        "get_date",
        "is_cloud",
        "push_workspace_file",
        "supported",
    }
    assert removed.isdisjoint(dir(Device))
    assert {
        "connect",
        "state",
        "supported_operations",
        "tap",
        "tap_element",
        "do",
        "expect",
    } <= set(dir(Device))


def test_legacy_root_exports_are_lazy_and_deprecated():
    import pytest

    import mobilerun
    from mobilerun.addons.agents import agents_available

    assert "MobileAgentState" not in vars(mobilerun)
    assert "MobileAgentState" not in mobilerun.__all__
    with pytest.warns(DeprecationWarning, match="mobilerun.MobileAgentState"):
        if agents_available():
            from mobilerun.agents import MobileAgentState

            assert mobilerun.MobileAgentState is MobileAgentState
        else:
            with pytest.raises(
                ModuleNotFoundError, match="pip install mobilerun-agents"
            ):
                _ = mobilerun.MobileAgentState


def test_workspace_methods_are_classmethods():
    from mobilerun import Workspace

    expected = {
        "upload_file",
        "list_files",
        "download_file",
        "delete_file",
        "upload_app",
        "list_apps",
        "get_app",
        "delete_app",
        "push_file",
    }
    assert {name for name in vars(Workspace) if not name.startswith("_")} == expected
    assert all(
        isinstance(inspect.getattr_static(Workspace, name), classmethod)
        for name in expected
    )
