"""Device connection, cleanup, pooled operations and timeout contracts."""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.devices import (
    AndroidDevice,
    DevicePool,
    IOSDevice,
    RecordingDevice,
    RemoteDevice,
    StealthDevice,
)
from mobilerun.core.devices.base import Device
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.devices.components.cloud_lifecycle import _CloudLifecycle
from mobilerun.core.mailbox.cloud import CloudMailbox
from mobilerun.core.operator import Operator
from mobilerun.models.apps import App
from mobilerun.models.results import ExecutionResult, ExpectationVerdict

URL = "http://alice:secret-password@localhost:5555/path?token=hidden"


def test_cloud_install_verifies_system_apps():
    device = CloudDevice(device_id="device-id", api_key="test")
    device._client = SimpleNamespace(
        devices=SimpleNamespace(
            apps=SimpleNamespace(
                install=AsyncMock(),
                list_installs=AsyncMock(
                    side_effect=[
                        SimpleNamespace(installs=[]),
                        SimpleNamespace(
                            installs=[
                                SimpleNamespace(
                                    app_id="pkg", status="succeeded", started_at=1
                                )
                            ]
                        ),
                    ]
                ),
            )
        )
    )

    async def inventory(include_system=False):
        return [{"package_name": "pkg", "version_code": 4}] if include_system else []

    device.list_apps = inventory
    app = App("pkg", "1", "Example", id="app-id", version_code=4)
    assert asyncio.run(device.install_app(app, timeout=0.05)) is app


@pytest.mark.parametrize("mode", ["direct", "wrapper", "pool", "implicit"])
def test_remote_connection_is_serialized(mode):
    async def run():
        device = RemoteDevice("http://localhost:5555")
        calls = 0

        async def connect():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            device._driver._connected = True

        device._driver.connect = connect
        if mode == "direct":
            await asyncio.gather(device.connect(), device.connect())
        elif mode == "wrapper":
            wrapped = RecordingDevice(device)
            await asyncio.gather(wrapped.connect(), wrapped.connect())
        elif mode == "pool":
            await DevicePool([RecordingDevice(device), StealthDevice(device)]).connect()
        else:
            await asyncio.gather(device.connect(), device._ensure_connected())
        assert calls == 1

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["remote", "legacy", "portal-http"])
def test_backend_log_redacts_url(kind, monkeypatch, caplog):
    from mobilerun.core.devices import ios

    caplog.set_level(logging.INFO)

    async def emit(*args, **kwargs):
        logging.getLogger("mobilerun_core_local").info("Connected to %s", URL)
        logging.getLogger("mobilerun_core_local").info("Connected to " + URL)

    async def run():
        if kind == "remote":
            device = RemoteDevice(URL)
            monkeypatch.setattr(device._driver, "connect", emit)
        else:
            device = IOSDevice(URL)
            monkeypatch.setattr(
                ios, "detect_ios_portal_kind", AsyncMock(return_value=kind)
            )
            monkeypatch.setattr(
                ios.IOSPortalDriver if kind == "legacy" else ios.IOSPortalHttpDriver,
                "ensure_connected",
                emit,
            )
        await device.connect()
        logging.getLogger("mobilerun.core.devices.test").info(
            "Connected to %s", device.identifier
        )

    asyncio.run(run())
    assert caplog.records
    for record in caplog.records:
        assert "alice" not in record.getMessage()
        assert "secret-password" not in record.getMessage()
        assert "hidden" not in record.getMessage()


@pytest.mark.parametrize(
    "mode", ["connect-button", "connect-screenshot", "first-ops", "wrappers", "pool"]
)
def test_public_operations_share_connection_lock(mode, monkeypatch):
    async def run():
        device = RemoteDevice("http://localhost:5555")
        calls = 0

        async def connect():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            device._driver._connected = True

        async def button(_):
            await device._driver.ensure_connected()

        async def screenshot(_=True):
            await device._driver.ensure_connected()
            return b"image"

        async def tap(*_):
            await device._driver.ensure_connected()

        monkeypatch.setattr(device._driver, "connect", connect)
        monkeypatch.setattr(device._driver, "press_button", button)
        monkeypatch.setattr(device._driver, "screenshot", screenshot)
        monkeypatch.setattr(device._driver, "tap", tap)
        if mode == "connect-button":
            work = asyncio.gather(device.connect(), device.press_button("home"))
        elif mode == "connect-screenshot":
            work = asyncio.gather(device.connect(), device.screenshot())
        elif mode == "first-ops":
            work = asyncio.gather(device.tap(1, 2), device.screenshot())
        elif mode == "wrappers":
            nested = RecordingDevice(StealthDevice(device))
            other = RecordingDevice(device)
            work = asyncio.gather(nested.press_button("home"), other.screenshot())
        else:
            pool = DevicePool([RecordingDevice(device), StealthDevice(device)])
            work = asyncio.gather(pool.press_button("home"), pool.screenshot())
        await asyncio.wait_for(work, timeout=1)
        assert calls == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "terminate_error",
    [None, ValueError("terminate")],
    ids=["cleanup-succeeds", "cleanup-fails"],
)
def test_android_cleanup_after_extension_failure(terminate_error):
    device = AndroidDevice(serial="test")
    device._close_extensions = AsyncMock(side_effect=RuntimeError("detach"))
    device._started_emulator = True
    device._keep = False
    device.terminate = AsyncMock(side_effect=terminate_error)
    with pytest.raises(RuntimeError, match="detach"):
        asyncio.run(device.close())
    device.terminate.assert_awaited_once()


@pytest.mark.parametrize("kind", ["ios", "remote"])
def test_client_cleanup_after_extension_failure(kind):
    if kind == "ios":
        device = IOSDevice("http://localhost")
    else:
        device = RemoteDevice("http://localhost")
    client = SimpleNamespace(aclose=AsyncMock())
    device._driver = SimpleNamespace(_client=client)
    device._close_extensions = AsyncMock(side_effect=RuntimeError("detach"))
    with pytest.raises(RuntimeError, match="detach"):
        asyncio.run(device.close())
    client.aclose.assert_awaited_once()


def test_cloud_cleanup_after_extension_failure(monkeypatch):
    from mobilerun.core.devices.components import cloud_lifecycle

    device = object.__new__(_CloudLifecycle)
    device._close_extensions = AsyncMock(side_effect=RuntimeError("detach"))
    device._close_resources = AsyncMock()
    device._connect_lock = asyncio.Lock()
    device._owned = True
    device._keep = False
    device.device_id = "device"
    device._billing = "included"
    device._client = object()
    device._owned_client = SimpleNamespace(close=AsyncMock())
    client = device._owned_client
    terminate = AsyncMock()
    monkeypatch.setattr(cloud_lifecycle.devices, "terminate", terminate)
    with pytest.raises(RuntimeError, match="detach"):
        asyncio.run(device.close())
    device._close_resources.assert_awaited_once()
    terminate.assert_awaited_once()
    client.close.assert_awaited_once()


def test_list_mailboxes_preserves_address(monkeypatch):
    from mobilerun.core.mailbox import cloud

    monkeypatch.setattr(
        cloud.mailboxes,
        "list_mailboxes",
        AsyncMock(return_value=[SimpleNamespace(id="id", address="a@example.com")]),
    )
    result = asyncio.run(CloudMailbox.list(billing="included"))
    assert result[0].address == "a@example.com"


def test_apk_missing_optional_version_name(monkeypatch, tmp_path):
    from mobilerun.clients.local import apk

    monkeypatch.setattr(apk.shutil, "which", lambda _: "/aapt")
    monkeypatch.setattr(
        apk.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout="package: name='example' versionCode='1'\n"
        ),
    )
    assert apk.inspect_apk(tmp_path / "app.apk") == ("example", 1, "", "example")


def test_apk_inspection_timeout(monkeypatch, tmp_path):
    from mobilerun.clients.local import apk

    monkeypatch.setattr(apk.shutil, "which", lambda _: "/aapt")

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] > 0
        raise apk.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(apk.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        apk.inspect_apk(tmp_path / "app.apk")


def test_android_apk_inspection_runs_off_event_loop(monkeypatch, tmp_path):
    import threading

    from mobilerun.core.devices import android

    loop_thread = threading.get_ident()
    seen = []

    def inspect(path):
        seen.append(threading.get_ident())
        return ("example", 1, "", "Example")

    monkeypatch.setattr(android, "inspect_apk", inspect)
    device = AndroidDevice(serial="test")
    device.install_app = AsyncMock()
    asyncio.run(device.install_apk(tmp_path / "a.apk"))
    assert seen and seen[0] != loop_thread


def test_cloud_upload_does_not_reuse_unverified_binary(monkeypatch, tmp_path):
    from mobilerun.clients.cloud import apps

    apk = tmp_path / "app.apk"
    apk.write_bytes(b"new binary")
    monkeypatch.setattr(apps, "inspect_apk", lambda _: ("example", 1, "1", "Example"))
    item = SimpleNamespace(id="old", bundle_id="example", display_name="Example")
    sdk = SimpleNamespace(
        apps=SimpleNamespace(
            list=AsyncMock(
                return_value=SimpleNamespace(
                    items=[item], pagination=SimpleNamespace(has_next=False)
                )
            ),
            list_versions=AsyncMock(
                return_value=SimpleNamespace(
                    data=[
                        SimpleNamespace(
                            status="available", version_code=1, version_name="1"
                        )
                    ]
                )
            ),
            create_signed_upload_url=AsyncMock(
                return_value=SimpleNamespace(
                    app_id="new",
                    version_id="v",
                    r2_upload_urls=[
                        SimpleNamespace(
                            file_name="app.apk", r2_upload_url="https://example.test"
                        )
                    ],
                )
            ),
            confirm_upload=AsyncMock(),
            mark_failed=AsyncMock(),
        )
    )
    client = AsyncMock()
    client.__aenter__.return_value = sdk
    monkeypatch.setattr(apps, "cloud_client", lambda *a, **kw: client)
    http = AsyncMock()
    http.__aenter__.return_value = http
    http.put.return_value = SimpleNamespace(raise_for_status=lambda: None)
    monkeypatch.setattr(apps.httpx, "AsyncClient", lambda **kw: http)
    result = asyncio.run(apps.upload_app(apk, billing="included"))
    assert result.id == "new"
    http.put.assert_awaited_once()


def test_cloud_upload_preparation_runs_off_event_loop(monkeypatch, tmp_path):
    import threading

    from mobilerun.clients.cloud import apps

    path = tmp_path / "app.apk"
    path.write_bytes(b"apk")
    main_thread = threading.get_ident()
    seen = []

    def inspect(_):
        seen.append(threading.get_ident())
        return ("example", 1, "", "Example")

    monkeypatch.setattr(apps, "inspect_apk", inspect)
    client = AsyncMock()
    client.__aenter__.side_effect = RuntimeError("stop after preparation")
    monkeypatch.setattr(apps, "cloud_client", lambda *a, **kw: client)
    with pytest.raises(RuntimeError, match="stop after preparation"):
        asyncio.run(apps.upload_app(path, billing="included"))
    assert seen and seen[0] != main_thread


class FakeDevice(Device):
    platform = "fake"
    supported_operations = {"tap"}
    needs_initialization = False

    def __init__(self, value, gate=None, error=None):
        super().__init__()
        self.value = value
        self._operator = None
        self.gate = gate
        self.error = error
        self.finished = False
        self.calls = []

    @property
    def state(self):
        return self

    async def screenshot(self, hide_overlay=True):
        if self.gate:
            await self.gate(self.value)
        self.finished = True
        if self.error:
            raise self.error
        return self.value

    async def _ensure_connected(self):
        self.calls.append("connected")

    async def connect(self):
        await self._ensure_connected()

    async def close(self):
        pass

    async def refresh(self):
        return self.value

    async def do(self, instruction, **kwargs):
        return instruction, kwargs

    async def expect(self, condition, **kwargs):
        return condition, kwargs


def test_empty_and_invalid_concurrency():
    with pytest.raises(ValueError):
        DevicePool([])
    with pytest.raises(ValueError):
        DevicePool([FakeDevice(1)], concurrency=0)


def test_concurrent_order_and_properties():
    async def run():
        ready = asyncio.Event()
        seen = []

        async def gate(value):
            seen.append(value)
            if len(seen) == 2:
                ready.set()
            await ready.wait()

        devices = [FakeDevice(1, gate), FakeDevice(2, gate)]
        pool = DevicePool(devices)
        assert await asyncio.wait_for(pool.screenshot(), 1) == [1, 2]
        assert len(pool) == 2 and tuple(pool) == pool.devices == tuple(devices)
        assert pool.platform == ["fake", "fake"]
        assert pool.supported_operations == [{"tap"}, {"tap"}]
        assert await pool.state.refresh() == [1, 2]

    asyncio.run(run())


def test_errors_wait_and_annotate():
    async def run():
        failure = ValueError("bad")
        devices = [FakeDevice(1, error=failure), FakeDevice(2)]
        with pytest.raises(ExceptionGroup) as caught:
            await DevicePool(devices).screenshot()
        assert all(device.finished for device in devices)
        assert caught.value.exceptions == (failure,)
        assert "device 0: FakeDevice" in failure.__notes__
        assert "1 of 2 devices failed: screenshot" in str(caught.value)

    asyncio.run(run())


def test_operator_forwarding_and_context():
    async def run():
        devices = [FakeDevice(1), FakeDevice(2)]
        operator = object()
        async with DevicePool(devices) as pool:
            assert (
                await pool.do("go", output=int, max_steps=2, operator=operator)
                == [("go", {"output": int, "max_steps": 2, "operator": operator})] * 2
            )
            assert (
                await pool.expect("yes", operator=operator)
                == [("yes", {"operator": operator})] * 2
            )
        assert all(device.calls == ["connected"] for device in devices)

    asyncio.run(run())


def test_concurrency_limit():
    async def run():
        active = peak = 0

        async def gate(value):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

        pool = DevicePool([FakeDevice(i, gate) for i in range(5)], concurrency=2)
        assert await pool.screenshot() == list(range(5))
        assert peak == 2

    asyncio.run(run())


def test_context_closes_supported_devices():
    class Closable(FakeDevice):
        async def close(self):
            self.calls.append("closed")

    async def run():
        devices = [Closable(1), FakeDevice(2)]
        async with DevicePool(devices):
            pass
        assert devices[0].calls == ["connected", "closed"]
        assert devices[1].calls == ["connected"]

    asyncio.run(run())


class FakeOperator(Operator):
    def __init__(self):
        self.calls = []

    async def execute(self, device, instruction, *, output=None, max_steps=None):
        self.calls.append((device, instruction, output, max_steps))
        verdict = ExpectationVerdict(passed=True, reason="visible") if output else None
        return ExecutionResult(success=True, output=verdict, device="fake")


def test_device_expect_with_per_call_operator():
    device = AndroidDevice(serial="test", portal_mode="disabled")
    operator = FakeOperator()
    assert device._operator is None
    assert asyncio.run(device.expect("visible", operator=operator)).passed
    assert operator.calls[0][0] is device
    assert device._operator is None


@pytest.mark.parametrize("wrapper", [RecordingDevice, StealthDevice])
def test_wrapper_forwards_per_call_operator(wrapper):
    device = AndroidDevice(serial="test", portal_mode="disabled")
    wrapped = wrapper(device)
    operator = FakeOperator()
    assert asyncio.run(wrapped.do("go", operator=operator)).success
    assert asyncio.run(wrapped.expect("visible", operator=operator)).passed
    assert [call[0] for call in operator.calls] == [wrapped, wrapped]
    assert device._operator is None


def test_pool_real_android_devices_forward_per_call_operator(monkeypatch):
    async def _ensure_connected(self):
        pass

    monkeypatch.setattr(AndroidDevice, "_ensure_connected", _ensure_connected)
    devices = [AndroidDevice(serial=str(i), portal_mode="disabled") for i in range(2)]
    operator = FakeOperator()

    async def run():
        async with DevicePool(devices) as pool:
            assert all(
                result.success for result in await pool.do("go", operator=operator)
            )
            assert all(
                result.passed
                for result in await pool.expect("visible", operator=operator)
            )

    asyncio.run(run())
    assert [call[0] for call in operator.calls] == devices + devices
    assert all(device._operator is None for device in devices)


def test_cleanup_failure_does_not_replace_body_exception():
    original = RuntimeError("body failed")

    class BrokenClose(FakeDevice):
        async def close(self):
            raise RuntimeError("cleanup failed")

    async def run():
        async with DevicePool([BrokenClose(1)]):
            raise original

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(run())
    assert caught.value is original
    assert any("cleanup" in note.lower() for note in original.__notes__)


@pytest.mark.parametrize("wrapper", [RecordingDevice, StealthDevice])
@pytest.mark.parametrize("pending", ["_ready", "_pending_avd"])
def test_pool_initializes_wrapped_pending_devices(wrapper, pending):
    class Pending(FakeDevice):
        needs_initialization = True

        def __init__(self):
            super().__init__(1)
            setattr(self, pending, pending == "_pending_avd")

        async def connect(self):
            self.calls.append("initialized")

        async def close(self):
            self.calls.append("closed")

    async def run():
        device = Pending()
        wrapped = wrapper(device)
        async with DevicePool([wrapped]):
            assert device.calls == ["initialized"]
        assert device.calls == ["initialized", "closed"]

    asyncio.run(run())


def test_device_do_forwards_timeout():
    class FakeOperator(Operator):
        async def execute(self, device, instruction, **kwargs):
            return kwargs

    device = Device(operator=FakeOperator())
    assert asyncio.run(device.do("go", timeout=3))["timeout"] == 3


def test_pool_forwards_timeout():
    class TimeoutDevice:
        _operator = None

        async def do(self, instruction, **kwargs):
            return kwargs

        async def expect(self, condition, **kwargs):
            return kwargs

    pool = DevicePool([TimeoutDevice()])
    assert asyncio.run(pool.do("go", timeout=3))[0]["timeout"] == 3
    assert asyncio.run(pool.expect("ok", timeout=4))[0]["timeout"] == 4


def test_macro_timeout_cancels_replay(monkeypatch):
    from mobilerun.macro import operator as macro_module

    cancelled = []

    class Player:
        def __init__(self, device, **kwargs):
            pass

        async def replay_macro(self, *args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

    monkeypatch.setattr(macro_module, "MacroPlayer", Player)
    monkeypatch.setattr(macro_module, "load_macro", lambda path: [])
    with pytest.raises(TimeoutError):
        asyncio.run(macro_module.MacroOperator().execute(object(), "go", timeout=0.01))
    assert cancelled == [True]


@pytest.mark.parametrize("cancel", [False, True])
def test_cloud_device_do_stops_task_on_timeout_or_cancel(cancel):
    from contextlib import asynccontextmanager

    from mobilerun.core.operator.cloud import CloudTaskOperator
    from mobilerun.errors import CloudTaskTimeout

    async def run():
        started = asyncio.Event()

        class Tasks:
            run = AsyncMock(return_value=SimpleNamespace(id="task-1"))
            stop = AsyncMock()

            @property
            def with_streaming_response(self):
                return self

            @asynccontextmanager
            async def attach(self, task_id, **kwargs):
                yield self

            async def iter_lines(self):
                started.set()
                await asyncio.Event().wait()
                yield ""

        tasks = Tasks()

        class FakeCloudDevice(Device):
            @property
            def _is_cloud(self):
                return True

        device = FakeCloudDevice(operator=CloudTaskOperator(billing="included"))
        device._task_started_callback = None
        device.device_id = "device-1"
        device._cloud_client = SimpleNamespace(tasks=tasks)
        device._ensure_connected = AsyncMock()
        pending = asyncio.create_task(device.do("go", timeout=10 if cancel else 0.02))
        await asyncio.wait_for(started.wait(), 1)
        if cancel:
            pending.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else CloudTaskTimeout):
            await pending
        tasks.stop.assert_awaited_once_with("task-1")

    asyncio.run(run())


def test_wait_sleeps_once_and_records_duration(monkeypatch, tmp_path):
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    device = Device()
    recording = RecordingDevice(device, runs_dir=tmp_path, capture="actions")
    asyncio.run(recording.wait(2.5))
    sleep.assert_awaited_once_with(2.5)
    row = json.loads((recording.run_dir / "steps.jsonl").read_text())
    assert (row["action"], row["args"]) == ("wait", {"duration": 2.5})


def test_pool_waits_once_per_device_in_parallel(monkeypatch):
    entered = []

    async def sleep(seconds):
        entered.append(seconds)
        if len(entered) == 2:
            ready.set()
        await ready.wait()

    monkeypatch.setattr(asyncio, "sleep", sleep)

    async def run():
        nonlocal ready
        ready = asyncio.Event()
        return await DevicePool([Device(), Device()]).wait(3)

    ready = None
    assert asyncio.run(run()) == [None, None]
    assert entered == [3, 3]
