import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner

from mobilerun.cli.devices import device
from mobilerun.cli.devices.base import discovery, selection
from mobilerun.clients.local import login_store
from mobilerun.config.loader import ConfigLoader, OutdatedConfigError
from mobilerun.config.user_config import MobileConfig


@pytest.fixture(autouse=True)
def clean_target(monkeypatch):
    monkeypatch.delenv("ANDROID_SERIAL", raising=False)
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)


@pytest.fixture
def adb_phone(monkeypatch):
    def set_devices(**states):
        monkeypatch.setattr(
            discovery.adb,
            "list",
            AsyncMock(
                return_value=[
                    SimpleNamespace(serial=k, state=v) for k, v in states.items()
                ]
            ),
        )

    return set_devices


@pytest.mark.parametrize(
    "serial,explicit,expected",
    [
        ("missing", None, None),
        ("wanted", None, "wanted"),
        ("missing", "chosen", "chosen"),
    ],
)
def test_file_rm_respects_target_precedence(
    monkeypatch, adb_phone, serial, explicit, expected
):
    cfg = MobileConfig()
    cfg.device.serial = "saved"
    ConfigLoader.save(cfg)
    monkeypatch.setenv("ANDROID_SERIAL", serial)
    adb_phone(saved="device", wanted="device")
    connect = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(selection, "connect_target", connect)
    args = ["file", "rm", "/demo", "--yes"]
    result = CliRunner().invoke(
        device, args + (["--device", explicit] if explicit else [])
    )
    if expected is None:
        connect.assert_not_awaited()
        assert result.exit_code == 1 and serial in result.stderr
    else:
        assert result.exit_code == 0, result.output
        assert connect.await_args.args == (expected,)


@pytest.mark.parametrize("online", [True, False])
def test_only_online_adb_device_is_selected(monkeypatch, adb_phone, online):
    adb_phone(
        off="offline", locked="unauthorized", **({"on": "device"} if online else {})
    )
    connect = AsyncMock(return_value=AsyncMock())
    monkeypatch.setattr(selection, "connect_target", connect)
    result = CliRunner().invoke(device, ["file", "rm", "/demo", "--yes"])
    if online:
        assert result.exit_code == 0, result.output
        assert connect.await_args.args == ("on",)
    else:
        assert result.exit_code == 1 and "No devices found" in result.stderr
        connect.assert_not_awaited()


@pytest.mark.parametrize("args", [[], ["--json"], ["--cloud"]])
def test_bad_login_does_not_hide_local_state(adb_phone, tmp_path, monkeypatch, args):
    adb_phone(locked="unauthorized")
    path = tmp_path / "bad.json"
    path.write_text("{")
    monkeypatch.setattr(login_store, "credential_path", lambda: path)
    result = CliRunner().invoke(device, ["list", *args])
    assert "credentials" in result.stderr.lower()
    if "--cloud" in args:
        assert result.exit_code == 1
    elif "--json" in args:
        assert result.exit_code == 0
        assert json.loads(result.stdout)[0]["state"] == "unauthorized"
    else:
        assert result.exit_code == 0 and "unauthorized" in result.stdout


def test_ios_does_not_receive_saved_android_token(monkeypatch):
    cfg = MobileConfig()
    cfg.device.auth_token = "android-test-token"
    create = Mock(return_value=AsyncMock())
    monkeypatch.setattr(selection, "IOSDevice", create)
    asyncio.run(selection.connect_target("http://ios", kind="ios", config=cfg))
    create.assert_called_once_with("http://ios", token=None)


def test_android_ping_does_not_receive_saved_ios_token(monkeypatch):
    from importlib import import_module

    ping = import_module("mobilerun.cli.devices.ping")
    cfg = MobileConfig()
    cfg.device.platform = "ios"
    cfg.device.auth_token = "ios-test-token"
    ConfigLoader.save(cfg)
    create = Mock(return_value=AsyncMock())
    monkeypatch.setattr(ping, "create_android_portal_http_driver", create)
    result = CliRunner().invoke(device, ["ping", "--android", "http://portal"])
    assert result.exit_code == 0, result.output
    create.assert_called_once_with("http://portal", None)


def test_missing_env_config_never_uses_global(tmp_path, monkeypatch):
    ConfigLoader.save(MobileConfig())
    monkeypatch.setenv("MOBILERUN_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(OutdatedConfigError, match="missing.yaml"):
        ConfigLoader.load()


@pytest.mark.parametrize("value", ["", " \t "])
def test_blank_env_config_uses_saved_user_config(monkeypatch, value):
    config = MobileConfig()
    config.device.serial = "saved"
    ConfigLoader.save(config)
    monkeypatch.setenv("MOBILERUN_CONFIG", value)
    assert ConfigLoader.load().device.serial == "saved"


def test_saved_token_and_env_override(monkeypatch):
    cfg = MobileConfig().device
    cfg.platform = "ios"
    cfg.auth_token = "ios-test-token"
    assert cfg.resolve_auth_token("android") is None
    assert cfg.resolve_auth_token("ios") == "ios-test-token"
    monkeypatch.setenv("MOBILERUN_DEVICE_TOKEN", "test-override")
    assert cfg.resolve_auth_token("android") == "test-override"
