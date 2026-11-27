"""Regressions from the device-contract review, with no live resources."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.devices import (
    AndroidDevice,
    CloudDevice,
    Device,
    IOSDevice,
    RecordingDevice,
    RemoteDevice,
    StealthDevice,
)
from mobilerun.core.operator import Operator
from mobilerun.core.proxy import (
    CloudFixedProxy,
    CloudRotatingProxy,
    CloudSocks5Proxy,
    LocalProxy,
)
from mobilerun.core.proxy.target import ProxyTarget
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.logs.actions import agent_line
from mobilerun.models import Billing, WorkspaceFile
from mobilerun.models.results import ExecutionResult, ExpectationVerdict

URL = "https://alice:secret-password@localhost:6643/path?token=hidden#hidden"
SAFE_URL = "https://localhost:6643/path"


@pytest.mark.parametrize("factory", [IOSDevice, AndroidDevice, RemoteDevice])
def test_url_identifiers_and_action_logs_do_not_leak_credentials(factory):
    device = factory(URL)
    assert device.identifier == SAFE_URL
    line = agent_line(
        SimpleNamespace(
            driver=device, ui=None, shared_state=SimpleNamespace(step_number=1)
        ),
        "tap",
        {"x": 1, "y": 2},
    )
    assert SAFE_URL in line
    assert all(
        secret not in line for secret in ("alice", "secret-password", "token", "hidden")
    )


@pytest.mark.parametrize("kind", ["portal-http", "legacy"])
def test_both_ios_factory_paths_sanitize_identifiers(monkeypatch, kind):
    from mobilerun.core.devices import ios

    monkeypatch.setattr(ios, "detect_ios_portal_kind", AsyncMock(return_value=kind))
    monkeypatch.setattr(IOSDevice, "_ensure_connected", AsyncMock())

    async def run():
        device = await IOSDevice(URL)
        try:
            assert device.identifier == SAFE_URL
        finally:
            await device.close()

    asyncio.run(run())


class InputDevice(Device):
    def __init__(self, **kwargs):
        self.calls = []
        super().__init__(**kwargs)

    async def tap(self, x, y):
        self.calls.append(("tap", x, y))

    async def _input_text_raw(self, text, *args):
        self.calls.append(("text", text))
        return True

    async def swipe(self, *args):
        self.calls.append(("swipe",))

    async def _swipe_humanized(self, *args):
        self.calls.append(("humanized",))


class InputOperator(Operator):
    def __init__(self):
        self.devices = []

    def check(self, device):
        pass

    async def execute(self, device, instruction, *, output=None, **kwargs):
        self.devices.append(device)
        await device.tap(1, 2)
        await device.input_text("hello world")
        await device.swipe(1, 2, 3, 4)
        return ExecutionResult(
            success=True,
            device=type(device).__name__,
            output=ExpectationVerdict(passed=True, reason="ok") if output else None,
        )


@pytest.mark.parametrize("method", ["do", "expect"])
@pytest.mark.parametrize("mode", ["recording", "stealth", "nested"])
@pytest.mark.parametrize("selection", ["attached", "default", "explicit"])
def test_wrapper_operators_keep_outer_identity(
    monkeypatch, method, mode, selection, tmp_path
):
    operator = InputOperator()
    inner = InputDevice()
    monkeypatch.setattr(inner, "wait", AsyncMock(return_value=None))
    if selection == "attached":
        inner._operator = operator
    elif selection == "default":
        monkeypatch.setattr(inner, "_default_operator", lambda: operator)
    recording = RecordingDevice(inner, runs_dir=tmp_path, capture="actions")
    device = (
        recording
        if mode == "recording"
        else StealthDevice(recording if mode == "nested" else inner)
    )
    asyncio.run(
        getattr(device, method)(
            "test", **({"operator": operator} if selection == "explicit" else {})
        )
    )
    assert operator.devices == [device]
    if mode != "recording":
        assert ("humanized",) in inner.calls
        assert [call[1] for call in inner.calls if call[0] == "text"] == [
            "hello",
            " ",
            "world",
        ]
    if mode == "recording":
        lines = (recording.run_dir / "steps.jsonl").read_text().splitlines()
        first, *_, last = map(json.loads, lines)
        assert (first["action"], first["args"]) == ("tap", {"x": 1, "y": 2})
        assert last["action"] == "swipe"
    else:
        assert recording.run_dir is None


@pytest.mark.parametrize("wrapped", [False, True])
def test_workspace_push_uses_explicit_source_not_destination(monkeypatch, wrapped):
    from mobilerun import Workspace
    from mobilerun.clients.cloud import client, files

    phone = CloudDevice(
        "phone",
        api_key="device-key",
        base_url="https://device.test/v1",
        billing=Billing.PAID,
    )
    file = WorkspaceFile(
        "file", "image.png", "image/png", 3, datetime.now(timezone.utc)
    )
    source = AsyncMock()
    source.__aenter__.return_value = source
    clients = []

    def connect(api_key, base_url, *, billing):
        clients.append((api_key, base_url, billing))
        return source

    monkeypatch.setattr(client, "cloud_client", connect)
    download = AsyncMock(return_value=b"abc")
    monkeypatch.setattr(files, "download_file", download)

    async def push(path, dest):
        assert Path(path).read_bytes() == b"abc"
        assert Path(path).name == "image.png"
        return dest

    monkeypatch.setattr(phone, "push_file", push)
    device = RecordingDevice(phone) if wrapped else phone
    assert (
        asyncio.run(
            Workspace.push_file(
                file,
                device,
                "/sdcard/image.png",
                api_key="source-key",
                base_url="https://source.test/v1",
                billing=Billing.PAID,
            )
        )
        == "/sdcard/image.png"
    )
    download.assert_awaited_once_with(source, "file", billing=Billing.PAID)
    assert clients == [("source-key", "https://source.test/v1", Billing.PAID)]
    source.__aexit__.assert_awaited_once()


def test_url_sanitizer_preserves_non_url_ids_and_ipv6():
    from mobilerun.core.devices.components.identifiers import _sanitize_identifier

    assert _sanitize_identifier("emulator-5554") == "emulator-5554"
    assert _sanitize_identifier(None) is None
    assert (
        _sanitize_identifier("https://user:pass@[::1]:6643/path?token=x#secret")
        == "https://[::1]:6643/path"
    )
    assert _sanitize_identifier("http://user:pass@[bad") == "invalid-url"


def test_visual_remote_retains_meaningful_device_ids():
    device = RemoteDevice(URL)
    device._driver.device_id = "remote-phone"
    assert device.identifier == "remote-phone"
    device._driver.device_id = URL
    assert device.identifier == SAFE_URL


@pytest.mark.parametrize(
    "proxy",
    [
        CloudRotatingProxy("US"),
        CloudFixedProxy("id"),
        CloudSocks5Proxy("host", 1080),
        LocalProxy("host", 8080),
    ],
)
def test_neutral_proxy_target_error(proxy):
    with pytest.raises(
        DeviceOperationNotSupported,
        match=f"{type(proxy).__name__} isn't supported on this device",
    ):
        asyncio.run(proxy._apply(ProxyTarget()))


def test_managed_proxies_use_distinct_hooks():
    class Target(ProxyTarget):
        connect_rotating = AsyncMock()
        connect_fixed = AsyncMock()

    target = Target()
    asyncio.run(CloudRotatingProxy("DE", False)._apply(target))
    target.connect_rotating.assert_awaited_once_with(country="DE", smart_ip=False)
    asyncio.run(CloudFixedProxy("p")._apply(target))
    target.connect_fixed.assert_awaited_once_with(id="p")


def test_base_phone_binding_is_an_unsupported_operation():
    assert not hasattr(Device(), "_bind_phone_number")


@pytest.mark.parametrize(
    "method,args",
    [
        ("push_file", ("file",)),
        ("list_files", ("/",)),
        ("pull_file", ("remote", "local")),
        ("delete_file", ("file",)),
        ("set_location", (1, 2)),
        ("reset_location", ()),
        ("set_timezone", ("UTC",)),
        ("open_url", ("https://example.com",)),
        ("install_apk", ("file",)),
        ("install_app", (None,)),
        ("_ensure_connected", ()),
        ("screenshot", ()),
        ("tap", (1, 2)),
        ("swipe", (1, 2, 3, 4)),
        ("start_app", ("p",)),
        ("stop_app", ("p",)),
        ("uninstall_app", ("p",)),
        ("list_apps", ()),
        ("press_button", ("home",)),
        ("press_key_code", (1,)),
        ("drag", (1, 2, 3, 4)),
        ("_get_ui_tree", ()),
        ("_get_date", ()),
    ],
)
def test_base_unsupported_message_format(method, args):
    with pytest.raises(DeviceOperationNotSupported) as error:
        asyncio.run(getattr(Device(), method)(*args))
    assert str(error.value) == f"Device doesn't support {method}"


@pytest.mark.parametrize("cls", [Device, AndroidDevice, CloudDevice, RecordingDevice])
@pytest.mark.parametrize("method", ["install_app", "install_apk"])
def test_install_options_are_typed(cls, method):
    from typing import Unpack, get_origin, get_type_hints

    annotation = get_type_hints(getattr(cls, method))["kwargs"]
    assert get_origin(annotation) is Unpack
    from mobilerun.core.devices.base import InstallOptions

    assert get_type_hints(InstallOptions) == {
        "timeout": float,
        "reinstall": bool,
        "grant_permissions": bool,
    }


@pytest.mark.parametrize("property_name", ["number", "mailbox"])
def test_base_resource_unsupported_message_format(property_name):
    with pytest.raises(DeviceOperationNotSupported) as error:
        getattr(Device(), property_name)
    expected = {
        "number": "No sms extension is bound",
        "mailbox": "No email extension is bound",
    }
    assert str(error.value) == expected[property_name]
