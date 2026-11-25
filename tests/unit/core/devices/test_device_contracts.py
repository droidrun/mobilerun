"""Device construction, portable operations, backend delegation, and example contracts."""

import asyncio
import importlib
import subprocess
import sys
from operator import attrgetter
from types import SimpleNamespace
from typing import assert_type, cast
from unittest.mock import AsyncMock

import pytest
from mobilerun_core_local.driver.android import AndroidDriver
from mobilerun_core_local.driver.base import DeviceDriver, PlatformUnsupportedError
from mobilerun_core_local.driver.ios import IOSDriver
from mobilerun_core_local.driver.visual_remote import VisualRemoteDriver

from mobilerun.core.credentials import DictCredentials
from mobilerun.core.devices import (
    AndroidDevice,
    Device,
    IOSDevice,
    RecordingDevice,
    RemoteDevice,
    StealthDevice,
)
from mobilerun.core.devices.components.wrapper import DeviceWrapper, DriverDevice
from mobilerun.core.devices.pool import DevicePool
from mobilerun.core.state.base import StateProvider, _should_resize_model_screenshot
from mobilerun.core.state.elements import _parse_bounds
from mobilerun.core.state.options import (
    StateProviderOptions,
    android_state_provider,
    screenshot_state_provider,
)
from mobilerun.core.state.ui_state import UIState
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.models import CloudCredentialReference, CloudTaskInputs
from tests.support.devices import MemoryDevice


@pytest.mark.parametrize(
    ("name", "driver", "args", "kwargs", "platform"),
    [
        (
            "AndroidDevice",
            AndroidDriver,
            (),
            {"serial": "test", "portal_mode": "disabled"},
            "Android",
        ),
        ("IOSDevice", IOSDriver, ("http://localhost:8080",), {}, "iOS"),
        (
            "RemoteDevice",
            VisualRemoteDriver,
            (),
            {"url": "http://localhost:8090", "device_id": "test"},
            "VisualRemote",
        ),
    ],
)
def test_device_constructor_and_driver_compatibility(
    name, driver, args, kwargs, platform, monkeypatch
):
    devices = importlib.import_module("mobilerun.core.devices")
    device_class = getattr(devices, name)
    assert issubclass(device_class, devices.Device)
    device = device_class(*args, **kwargs)
    if name == "IOSDevice":

        monkeypatch.setattr(
            "mobilerun.core.devices.ios.detect_ios_portal_kind",
            AsyncMock(return_value="legacy"),
        )
        monkeypatch.setattr(device, "_ensure_connected", AsyncMock())
        assert device._driver is None
        asyncio.run(device.connect())
        platform = "ios"
    assert isinstance(device._driver, driver)
    assert not isinstance(device, DeviceDriver)
    assert device.platform == platform
    for wrapper in (devices.RecordingDevice, devices.StealthDevice):
        wrapped = wrapper(device)
        assert wrapped._inner is device
        assert wrapped.platform == platform


def test_cloud_device_is_lazy_and_other_devices_do_not_require_cloud():
    script = """
import importlib.abc
import sys
class BlockCloud(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mobilerun_core_local.driver.cloud":
            raise ImportError("cloud deliberately unavailable")
sys.meta_path.insert(0, BlockCloud())
import mobilerun.core.devices as devices
assert "mobilerun_core_local.driver.cloud" not in sys.modules
assert devices.AndroidDevice
try:
    devices.CloudDevice
except ImportError as exc:
    assert "cloud deliberately unavailable" in str(exc)
else:
    raise AssertionError("cloud should be imported only on access")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_cloud_device_constructor_and_driver_compatibility():
    cloud = pytest.importorskip("mobilerun_core_local.driver.cloud")
    devices = importlib.import_module("mobilerun.core.devices")
    cls = devices.CloudDevice
    assert issubclass(cls, devices.Device)
    device = cls("test-device", api_key="test-key")
    assert isinstance(device._driver, cloud.CloudDriver)
    assert not isinstance(device, DeviceDriver)
    assert device.device_id == "test-device"
    assert device.platform == "Android"
    for wrapper in (devices.RecordingDevice, devices.StealthDevice):
        assert wrapper(device)._inner is device


@pytest.mark.parametrize(
    "name",
    ["set_clipboard", "get_clipboard", "set_language"],
)
def test_environment_surface_is_portable(name: str) -> None:
    from mobilerun.core.devices.cloud import CloudDevice

    for cls in (Device, DeviceWrapper, RecordingDevice, StealthDevice, DevicePool):
        assert getattr(cls, name) is not None
    assert getattr(CloudDevice, name) is not None


def test_wrappers_preserve_typed_backend_handle() -> None:
    from mobilerun.core.devices.cloud import CloudDevice

    cloud = CloudDevice("phone", api_key="test")
    wrapped = RecordingDevice(StealthDevice(cloud))
    assert_type(wrapped._inner._inner, CloudDevice)
    assert wrapped._inner._inner is cloud
    assert wrapped._unwrap() is cloud


def test_local_number_and_unbound_mailbox_explain_unavailability() -> None:
    with pytest.raises(DeviceOperationNotSupported, match="No sms extension"):
        _ = Device().number
    with pytest.raises(DeviceOperationNotSupported, match="No email extension"):
        _ = Device().mailbox


def test_cloud_resources_work_through_wrappers_and_pool(monkeypatch) -> None:
    from datetime import datetime, timezone

    from mobilerun.clients.cloud import devices, mailboxes, numbers
    from mobilerun.core.devices.cloud import CloudDevice
    from mobilerun.core.mailbox import CloudMailbox
    from mobilerun.core.phone_number import CloudPhoneNumber

    async def run() -> None:
        phone = CloudDevice("phone", api_key="test")
        phone._cloud_telephony = True
        number = CloudPhoneNumber("number", api_key="test")
        mailbox = CloudMailbox("mailbox", api_key="test")
        monkeypatch.setattr(devices, "bind_number", AsyncMock())
        await number.bind(phone)
        await mailbox.bind(phone)
        now = datetime.now(timezone.utc)
        sms = {
            "id": "message",
            "createdAt": now,
            "occurredAt": now,
            "direction": "inbound",
            "status": "received",
            "detectedSender": "sender",
            "body": "Code 123456",
        }
        monkeypatch.setattr(
            numbers,
            "messages",
            AsyncMock(return_value=[numbers.MessageItem.model_validate(sms)]),
        )
        result = numbers.MessageSendResponse.model_validate({"data": sms})
        monkeypatch.setattr(numbers, "send", AsyncMock(return_value=result))
        monkeypatch.setattr(
            mailboxes,
            "otp",
            AsyncMock(
                return_value=SimpleNamespace(data=SimpleNamespace(code="654321"))
            ),
        )
        wrapper = DeviceWrapper(phone)
        assert await wrapper.number.wait_for_code() == "123456"
        assert len(await wrapper.number.messages()) == 1
        assert await wrapper.number.send("recipient", "message") == result
        assert await wrapper.mailbox.wait_for_code() == "654321"
        pool = DevicePool([wrapper])
        assert pool.number == [number]
        assert pool.mailbox == [mailbox]

    asyncio.run(run())


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("tap", (1, 2)),
        ("swipe", (1, 2, 3, 4)),
        ("drag", (1, 2, 3, 4)),
        ("input_text", ("hello",)),
        ("screenshot", ()),
        ("start_app", ("app",)),
        ("stop_app", ("app",)),
        ("uninstall_app", ("app",)),
        ("list_apps", ()),
        ("press_button", ("home",)),
        ("press_key_code", (3,)),
        ("_get_ui_tree", ()),
        ("_get_date", ()),
        ("_ensure_connected", ()),
    ],
)
def test_base_unsupported_operations_have_framework_error(method, args) -> None:
    with pytest.raises(DeviceOperationNotSupported):
        asyncio.run(getattr(Device(), method)(*args))


@pytest.mark.parametrize("error", [NotImplementedError, PlatformUnsupportedError])
@pytest.mark.parametrize("operation", ["tap", "stop_app", "_get_ui_tree"])
def test_raw_adapter_translates_unsupported(error, operation) -> None:
    class UnsupportedDriver(DeviceDriver):
        async def tap(self, x: int, y: int) -> None:
            raise error("backend unsupported")

        async def stop_app(self, package: str) -> str:
            raise error("backend unsupported")

        async def get_ui_tree(self) -> dict[str, object]:
            raise error("backend unsupported")

    device = DriverDevice(
        UnsupportedDriver(), state_provider_factory=screenshot_state_provider
    )
    args = {"tap": (1, 2), "stop_app": ("app",), "_get_ui_tree": ()}[operation]
    with pytest.raises(DeviceOperationNotSupported, match="backend unsupported"):
        asyncio.run(getattr(device, operation)(*args))


@pytest.mark.parametrize(
    "method,args",
    [("list_apps", ()), ("_get_date", ()), ("_get_ui_tree", ())],
)
def test_visual_remote_does_not_report_empty_unsupported_results(method, args) -> None:

    with pytest.raises(DeviceOperationNotSupported):
        asyncio.run(getattr(RemoteDevice("http://localhost:1"), method)(*args))


@pytest.mark.parametrize(
    "device",
    [
        AndroidDevice("fake"),
        IOSDevice("http://localhost"),
        RemoteDevice("http://localhost"),
    ],
)
def test_root_io_uses_its_backend_and_live_capabilities(device):
    if device._driver is None:
        from mobilerun_core_local.driver.ios import IOSPortalDriver

        device._driver = IOSPortalDriver("http://localhost")
    device._driver.tap = AsyncMock()
    device._driver.ensure_connected = AsyncMock()
    device._driver.supported = {"tap"}
    asyncio.run(device.tap(4, 8))
    device._driver.tap.assert_awaited_once_with(4, 8)
    assert "tap" in device.supported_operations


def test_cloud_io_syncs_created_id_and_current_clients():
    from mobilerun.core.devices.cloud import CloudDevice

    device = CloudDevice(api_key="test")
    device.device_id = "created"
    device._ready = True
    tap = AsyncMock()
    device._write_client = SimpleNamespace(
        devices=SimpleNamespace(actions=SimpleNamespace(tap=tap))
    )
    asyncio.run(device.tap(4, 8))
    assert tap.await_args.args == ("created",)
    assert device._driver._client is device._client
    assert device._driver._write_client is device._write_client


def test_pending_android_action_never_reaches_raw_backend():
    device = AndroidDevice(avd="fake")
    device._driver.tap = AsyncMock()
    with pytest.raises(RuntimeError, match="Await AndroidDevice"):
        asyncio.run(device.tap(4, 8))
    device._driver.tap.assert_not_awaited()


@pytest.mark.parametrize("value", ["1,2,3", "1,2,3,4,5", "a,2,3,4"])
def test_bounds_parser_rejects_malformed_geometry(value):
    with pytest.raises(ValueError):
        _parse_bounds(value)


def test_bounds_parser_has_exact_four_coordinate_shape():
    assert _parse_bounds("1,2,3,4") == (1, 2, 3, 4)


def test_cloud_rejects_unknown_device_type_before_create():
    from mobilerun.core.devices.cloud import CloudDevice

    with pytest.raises(ValueError, match="Invalid cloud device type"):
        CloudDevice(api_key="test", device_type="not-a-device-type")


class CapabilityDevice(Device):
    @property
    def platform(self):
        return "Android"

    @property
    def supported_operations(self):
        return {"tap"}

    @property
    def supported_buttons(self):
        return {"home"}

    def __init__(self):
        super().__init__()
        self.connected = 0
        self.closed = 0
        self.typed = []
        self.taps = []
        self.recover = AsyncMock()

    @property
    def identifier(self):
        return "fake"

    @property
    def _state_recovery(self):
        return self.recover

    async def _ensure_connected(self):
        self.connected += 1

    async def close(self):
        self.closed += 1

    async def _input_text_raw(self, text, clear=False, stealth=False, wpm=0):
        self.typed.append(text)
        return True

    async def tap(self, x, y):
        self.taps.append((x, y))

    async def swipe(self, x1, y1, x2, y2, duration_ms=1000):
        pass

    def _create_state_provider(self, options=None):
        return android_state_provider(self, options or StateProviderOptions())


@pytest.mark.parametrize(
    "wrappers", [(RecordingDevice, StealthDevice), (StealthDevice, RecordingDevice)]
)
def test_nested_lifecycle_await_is_outer_and_cleanup_is_once(wrappers):
    root = CapabilityDevice()
    wrapped = wrappers[0](wrappers[1](root))

    async def run():
        assert await wrapped is wrapped
        assert await wrapped is wrapped
        async with wrapped as entered:
            assert entered is wrapped

    asyncio.run(run())
    assert root.connected == 3
    assert root.closed == 1
    assert wrapped._unwrap() is root
    assert wrapped.identifier == "fake"
    assert wrapped._state_recovery is root.recover


@pytest.mark.parametrize(
    "wrappers", [(RecordingDevice, StealthDevice), (StealthDevice, RecordingDevice)]
)
def test_vault_recording_and_raw_secret_typing_in_both_orders(wrappers):

    root = CapabilityDevice()
    wrapped = wrappers[0](wrappers[1](root))
    wrapped.state.refresh = AsyncMock(
        return_value=SimpleNamespace(
            phone_state={
                "focusedElement": {"className": "EditText"},
                "isEditable": True,
            }
        )
    )

    async def run():
        vault = await DictCredentials(
            {"app": {"work": {"password": "literal{{secret}}"}}}
        ).bind(wrapped)
        await vault.type_field("app", "password", name="work")
        await wrapped.swipe(1, 2, 3, 4)

    asyncio.run(run())
    assert root.typed == ["literal{{secret}}"]
    assert len(wrapped.extensions) == 1
    assert wrapped._recording is not None
    assert [a["action_type"] for a in wrapped._recording.recorded_actions] == [
        "extension_call",
        "swipe",
    ]
    assert "literal" not in repr(wrapped._recording.recorded_actions)


def test_raw_adapter_requires_pairing_and_does_not_infer_close(monkeypatch):
    raw = DeviceDriver()
    raw.input_text = AsyncMock(return_value=True)
    raw.connect = AsyncMock()
    close = AsyncMock()
    monkeypatch.setattr(raw, "close", close, raising=False)
    with pytest.raises(TypeError, match="state_provider_factory"):
        DriverDevice(raw, **{})
    device = DriverDevice(raw, state_provider_factory=android_state_provider)
    asyncio.run(device.connect())
    asyncio.run(device.close())
    close.assert_not_awaited()
    assert device._unwrap() is device
    assert device._driver is raw
    assert device._state_recovery is None
    assert asyncio.run(device._input_coordinate_size(100, 200)) == (100, 200)


def test_provider_flags_defaults_and_resize_or_contract():
    provider = StateProvider(CapabilityDevice())
    assert not provider.use_normalized
    assert not provider.requires_active_contract_for_coords
    assert not _should_resize_model_screenshot(provider)
    provider.requires_coordinate_tools = True
    assert _should_resize_model_screenshot(provider)


def test_android_connection_failure_cleans_up_started_avd(monkeypatch):

    device = AndroidDevice(avd="fake")

    async def boot():
        device._pending_avd = False
        device._started_emulator = True

    device._boot = boot
    device._ensure_connected = AsyncMock(side_effect=RuntimeError("connection failed"))
    device.terminate = AsyncMock()
    with pytest.raises(RuntimeError, match="connection failed"):
        asyncio.run(device.connect())
    device.terminate.assert_awaited_once()


@pytest.mark.parametrize("wrapper", [RecordingDevice, StealthDevice])
def test_android_recovery_through_wrapper_restores_declared_keyboard(
    monkeypatch, wrapper
):

    device = AndroidDevice("fake")
    adb = SimpleNamespace(
        shell=AsyncMock(),
        shell2=AsyncMock(return_value=SimpleNamespace(returncode=0, output="null")),
    )
    monkeypatch.setattr(device, "_adb_device", adb)
    device._portal = SimpleNamespace(tcp_available=False)
    device._portal_keyboard_available = True
    setup = AsyncMock()
    monkeypatch.setattr(
        "mobilerun_core_local.driver.android.portal.setup_keyboard", setup
    )
    monkeypatch.setattr("mobilerun.core.devices.android.asyncio.sleep", AsyncMock())
    recovery = wrapper(device)._state_recovery
    assert recovery is not None
    asyncio.run(recovery())
    setup.assert_awaited_once_with(device._adb_device)
    assert adb.shell.await_count == 2


def test_provider_and_factory_attachment_are_separate_and_lazy():
    root = CapabilityDevice()

    def factory(device):
        return StateProvider(device)

    root._set_state_provider_factory(factory)
    assert root._ui_state_view._provider is None
    provider = root.state._resolved_provider
    assert provider._device is root
    root._set_state_provider(None)
    assert root.state._resolved_provider is not provider
    assert root._create_state_provider(
        StateProviderOptions(normalized=True)
    ).use_normalized
    with pytest.raises(TypeError, match="state_provider_factory"):
        root._set_state_provider(cast(StateProvider, factory))


def test_view_exposes_snapshot_not_unknown_provider_members():
    root = MemoryDevice()
    provider = StateProvider(root)
    state = UIState([], "snapshot", "", {}, 100, 200)
    provider.get_state = AsyncMock(return_value=state)
    root._set_state_provider(provider)
    asyncio.run(root.state.refresh())
    assert root.state.current.formatted_text == "snapshot"
    assert root.state.current.screen_width == 100
    with pytest.raises(AttributeError):
        attrgetter("unknown_provider_member")(root.state)


def test_every_declared_raw_method_is_explicitly_delegated():
    root = CapabilityDevice()
    wrapped = DeviceWrapper(root)

    async def run():
        calls = [
            ("connect", ()),
            ("_ensure_connected", ()),
            ("screenshot", (False,)),
            ("tap", (1, 2)),
            ("swipe", (1, 2, 3, 4, 1000)),
            ("drag", (1, 2, 3, 4, 3)),
            ("press_button", ("home",)),
            ("press_key_code", (4,)),
            ("start_app", ("app", None)),
            ("stop_app", ("app",)),
            ("uninstall_app", ("app",)),
            ("list_apps", (True,)),
            ("_get_ui_tree", ()),
            ("_get_date", ()),
            ("wait", (1,)),
            ("_input_coordinate_size", (10, 20)),
        ]
        for name, args in calls:
            mock = AsyncMock(return_value=None)
            # Test instrumentation, not production capability detection.
            setattr(root, name, mock)
            await getattr(wrapped, name)(*args)
            mock.assert_awaited_once_with(*args)

    asyncio.run(run())


def test_local_credentials_fail_closed_even_empty():
    assert DictCredentials({}).cloud_native is False


@pytest.mark.parametrize(
    "kind,expected",
    [("legacy", "IOSStateProvider"), ("portal-http", "AndroidStateProvider")],
)
def test_ios_factory_detects_once_and_pairs_point_coordinates(
    monkeypatch, kind, expected
):

    detect = AsyncMock(return_value=kind)
    monkeypatch.setattr("mobilerun.core.devices.ios.detect_ios_portal_kind", detect)
    monkeypatch.setattr(IOSDevice, "_ensure_connected", AsyncMock())
    device = IOSDevice("http://localhost:8080", token="fake")
    asyncio.run(device.connect())
    asyncio.run(device.connect())
    assert isinstance(device, IOSDevice)
    assert type(device._create_state_provider()).__name__ == expected
    assert device._state_recovery is None
    assert device.identifier == "http://localhost:8080"
    detect.assert_awaited_once()
    device._driver.input_coordinate_size = AsyncMock(return_value=(390, 844))
    assert asyncio.run(device._input_coordinate_size(1170, 2532)) == (390, 844)
    asyncio.run(device.close())


def test_cloud_empty_vault_has_empty_reference_tuple(monkeypatch):
    from mobilerun.core.credentials import CloudCredentials

    provider = CloudCredentials(api_key="test")
    provider._list_credentials = AsyncMock(return_value=[])
    assert asyncio.run(provider.cloud_task_inputs()).credential_references == ()


@pytest.mark.parametrize("mixed", [False, True])
def test_cloud_operator_reference_capability_dedupes_and_rejects_local(
    monkeypatch, mixed
):
    from mobilerun.core.devices import CloudDevice
    from mobilerun.core.operator.cloud import CloudTaskOperator
    from mobilerun.errors import OperatorNotSupported
    from mobilerun.models.results import ExecutionResult

    class Vault(DictCredentials):
        cloud_native = True

        async def cloud_task_inputs(self):
            return CloudTaskInputs(
                (
                    CloudCredentialReference("app", "work"),
                    CloudCredentialReference("app", "work"),
                )
            )

    phone = CloudDevice("id", api_key="fake")
    run_task = AsyncMock(
        return_value=ExecutionResult(success=True, device="CloudDevice")
    )
    monkeypatch.setattr("mobilerun.core.operator.cloud.run_task", run_task)

    async def run():
        await Vault({}).bind(phone)
        if mixed:
            await DictCredentials({}, name="local").bind(phone)
            with pytest.raises(OperatorNotSupported):
                await CloudTaskOperator().execute(phone, "test")
            run_task.assert_not_awaited()
        else:
            await CloudTaskOperator().execute(phone, "test")
            assert run_task.await_args is not None
            assert run_task.await_args.kwargs["credentials"] == [
                {"package_name": "app", "credential_names": ["work"]}
            ]

    asyncio.run(run())
