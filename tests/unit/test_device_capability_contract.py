"""Framework capability behavior, with no device, network or paid calls."""

import asyncio
from operator import attrgetter
from typing import cast
from unittest.mock import AsyncMock

import pytest
from mobilerun_core_local.driver.base import DeviceDriver

from mobilerun.core.credentials import DictCredentials
from mobilerun.core.devices import Device, RecordingDevice, StealthDevice
from mobilerun.core.devices.components.wrapper import DeviceWrapper, DriverDevice
from mobilerun.core.state.base import StateProvider, _should_resize_model_screenshot
from mobilerun.core.state.options import StateProviderOptions, android_state_provider
from mobilerun.core.state.ui_state import UIState
from mobilerun.models import CloudCredentialReference, CloudTaskInputs


class FakeDevice(Device):
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
    root = FakeDevice()
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
    from types import SimpleNamespace

    root = FakeDevice()
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
    provider = StateProvider(FakeDevice())
    assert not provider.use_normalized
    assert not provider.requires_active_contract_for_coords
    assert not _should_resize_model_screenshot(provider)
    provider.requires_coordinate_tools = True
    assert _should_resize_model_screenshot(provider)


def test_android_connection_failure_cleans_up_started_avd(monkeypatch):
    from mobilerun.core.devices import AndroidDevice

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
    from types import SimpleNamespace

    from mobilerun.core.devices import AndroidDevice

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
    root = FakeDevice()

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
    root = FakeDevice()
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
    root = FakeDevice()
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
    from mobilerun.core.devices import IOSDevice

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
