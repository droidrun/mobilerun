"""Offline contracts for Android, emulator, and Portal HTTP backends."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from click.testing import CliRunner
from mobilerun_core_local.driver.android import AndroidPortalHttpDriver
from mobilerun_core_local.driver.android.portal import (
    PORTAL_PACKAGE_NAME,
    portal_a11y_service,
)

from mobilerun.cli.devices import device as cli
from mobilerun.cli.devices.base import selection
from mobilerun.clients.portal.http import (
    create_android_portal_http_driver,
    is_portal_http_url,
)
from mobilerun.clients.portal.state_fetch import fetch_state_with_retry
from mobilerun.config.settings import get_settings
from mobilerun.config.user_config import MobileConfig
from mobilerun.core.devices import AndroidDevice, DevicePool, android
from mobilerun.core.devices.components.wrapper import DriverDevice
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.operator import Operator
from mobilerun.core.state.options import android_state_provider
from mobilerun.errors import StateFetchFailed
from mobilerun.models.results import ExecutionResult
from tests.support.devices import MemoryDevice


@pytest.mark.parametrize(
    "shell_error", [None, RuntimeError()], ids=["success", "failure"]
)
def test_close_attempts_keyboard_cleanup(shell_error):
    device = AndroidDevice(serial="test")
    shell = AsyncMock(side_effect=shell_error)
    device._adb_device = SimpleNamespace(shell=shell)
    asyncio.run(device.close())
    shell.assert_awaited_once()
    assert shell.await_args.args[0].startswith("ime disable ")


def test_disabled_portal_has_no_recovery():
    assert AndroidDevice("fake", portal_mode="disabled")._state_recovery is None


def test_failed_adb_only_tree_read_does_not_touch_accessibility(monkeypatch):
    device = AndroidDevice("fake", portal_mode="auto")
    adb = SimpleNamespace(
        shell=AsyncMock(),
        shell2=AsyncMock(return_value=SimpleNamespace(returncode=0, output="null")),
    )
    device._adb_device = adb
    fetch = AsyncMock(side_effect=RuntimeError("ADB tree unavailable"))
    monkeypatch.setattr(android.asyncio, "sleep", AsyncMock())
    with pytest.raises(StateFetchFailed, match="ADB tree unavailable"):
        asyncio.run(
            fetch_state_with_retry(
                fetch,
                device._state_recovery,
                max_retries=2,
                retry_delays=[0],
                recovery_after=1,
            )
        )
    assert fetch.await_count == 2
    adb.shell.assert_not_awaited()
    adb.shell2.assert_not_awaited()


@pytest.mark.parametrize("current", ["reader/.Main:magnifier/.Main", "null", "already"])
def test_recovery_preserves_services(monkeypatch, current):
    device = AndroidDevice("fake")
    device._portal = SimpleNamespace(tcp_available=False)
    portal = portal_a11y_service(PORTAL_PACKAGE_NAME)
    if current == "already":
        current = f"reader/.Main:{portal}"
    read = AsyncMock(return_value=SimpleNamespace(output=current, returncode=0))
    shell = AsyncMock()
    device._adb_device = SimpleNamespace(shell=shell, shell2=read)
    monkeypatch.setattr(android.asyncio, "sleep", AsyncMock())
    asyncio.run(device._state_recovery())
    commands = [c.args[0] for c in shell.await_args_list]
    read.assert_awaited_once_with(
        "settings get secure enabled_accessibility_services", rstrip=True
    )
    expected = []
    if portal not in current.split(":"):
        merged = portal if current == "null" else f"{current}:{portal}"
        expected.append(f"settings put secure enabled_accessibility_services {merged}")
    expected.append("settings put secure accessibility_enabled 1")
    assert commands == expected


def test_recovery_does_not_write_after_failed_settings_read():
    device = AndroidDevice("fake")
    device._portal = SimpleNamespace(tcp_available=False)
    shell = AsyncMock()
    device._adb_device = SimpleNamespace(
        shell=shell,
        shell2=AsyncMock(
            return_value=SimpleNamespace(output="permission denied", returncode=1)
        ),
    )
    with pytest.raises(RuntimeError, match="read"):
        asyncio.run(device._state_recovery())
    shell.assert_not_awaited()


def test_close_waits_for_owned_emulator(monkeypatch):
    async def scenario():
        device = AndroidDevice(avd="test")
        entered, resume = asyncio.Event(), asyncio.Event()

        async def boot(*args, **kwargs):
            entered.set()
            await resume.wait()

        for name, result in [
            ("find_running", None),
            ("list_avds", ["test"]),
            ("start_avd", "fake"),
            ("stop", None),
        ]:
            monkeypatch.setattr(android.emulator, name, AsyncMock(return_value=result))
        monkeypatch.setattr(android.emulator, "wait_boot", boot)
        monkeypatch.setattr(device, "_ensure_connected", AsyncMock())
        startup = asyncio.create_task(device.connect())
        await entered.wait()
        closing = asyncio.create_task(device.close())
        await asyncio.sleep(0)
        resume.set()
        await asyncio.gather(startup, closing)
        android.emulator.stop.assert_awaited_once_with("fake")
        assert not device._started_emulator

    asyncio.run(scenario())


def test_closed_avd_cannot_start(monkeypatch):
    device = AndroidDevice(avd="test")
    boot = AsyncMock(side_effect=AssertionError("started after close"))
    monkeypatch.setattr(device, "_boot", boot)
    asyncio.run(device.close())
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(device.connect())
    boot.assert_not_awaited()


def test_pool_validates_execution_facade():
    class RecordingOperator(Operator):
        def check(self, device):
            assert isinstance(device, RecordingDevice)

        async def execute(self, device, *args, **kwargs):
            self.check(device)
            return ExecutionResult(success=True, device="fake")

    device = RecordingDevice(MemoryDevice(), operator=RecordingOperator())
    assert asyncio.run(DevicePool([device]).do("go"))[0].success


@pytest.mark.parametrize("suffix", [[], ["get"], ["set"]])
def test_clipboard_help_identifies_supported_targets(suffix):
    result = CliRunner().invoke(cli, ["input", "clipboard", *suffix, "--help"])
    assert result.exit_code == 0
    assert "cloud device" in result.output.lower().split("options:")[0]


@pytest.mark.parametrize("args", [["get"], ["set", "private text"]])
def test_local_clipboard_names_the_unsupported_operation(monkeypatch, args):
    from mobilerun.cli.devices.base import selection
    from mobilerun.config.loader import ConfigLoader
    from mobilerun.config.user_config import MobileConfig

    monkeypatch.setattr(ConfigLoader, "load", lambda path: MobileConfig())

    monkeypatch.setattr(
        selection, "connect_target", AsyncMock(return_value=AndroidDevice("fake"))
    )
    result = CliRunner().invoke(cli, ["input", "clipboard", *args, "-d", "fake"])
    assert result.exit_code == 1
    assert f"AndroidDevice doesn't support {args[0]}_clipboard" in result.output
    assert "private text" not in result.output


PORTAL_URL = "http://192.168.1.20:8080"


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("http://192.168.1.20:8080", True),
        ("HTTPS://phone.local:8080", True),
        ("emulator-5554", False),
        ("192.168.1.20:5555", False),
        ("", False),
        (None, False),
    ],
)
def test_is_portal_http_url(device, expected):
    assert is_portal_http_url(device) is expected


def test_portal_http_driver_requires_token():
    with pytest.raises(ValueError, match="MOBILERUN_DEVICE_TOKEN"):
        create_android_portal_http_driver(PORTAL_URL, None)


def test_portal_http_driver_normalizes_url():
    driver = create_android_portal_http_driver(PORTAL_URL + "/", "sekret")

    assert isinstance(driver, AndroidPortalHttpDriver)
    assert driver.url == PORTAL_URL
    assert driver.token == "sekret"


@pytest.fixture
def driver():
    return AndroidPortalHttpDriver(
        "http://portal.test",
        "test-token",
        transport=httpx.MockTransport(lambda _: httpx.Response(200)),
    )


@pytest.mark.anyio
async def test_cli_device_closes_owned_portal_client(driver, monkeypatch):
    monkeypatch.setattr(
        selection, "create_android_portal_http_driver", lambda *_: driver
    )
    device = await selection.connect_target(
        driver.url, kind="android", config=MobileConfig()
    )
    client = driver._client
    assert not client.is_closed
    await device.close()
    assert client.is_closed


@pytest.mark.anyio
async def test_driver_device_leaves_borrowed_portal_client_open(driver):
    device = DriverDevice(driver, state_provider_factory=android_state_provider)
    await device.connect()
    client = driver._client
    try:
        await device.close()
        assert not client.is_closed
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (None, ("android_cloud_phone", None)),
        (("custom-cloud", "Pixel_API"), ("custom-cloud", "Pixel_API")),
    ],
    ids=["defaults", "overrides"],
)
def test_device_type_settings(monkeypatch, overrides, expected):
    keys = ("MOBILERUN_CLOUD_DEVICE_TYPE", "MOBILERUN_ANDROID_DEVICE_TYPE")
    if overrides is None:
        for key in keys:
            monkeypatch.delenv(key, raising=False)
    else:
        for key, value in zip(keys, overrides, strict=True):
            monkeypatch.setenv(key, value)
    settings = get_settings()
    assert (
        settings.MOBILERUN_CLOUD_DEVICE_TYPE,
        settings.MOBILERUN_ANDROID_DEVICE_TYPE,
    ) == expected


def test_cloud_create_uses_setting_and_explicit_override(monkeypatch):
    from mobilerun.core.devices import cloud

    create = AsyncMock(return_value=SimpleNamespace(id="device", name="device"))
    devices = SimpleNamespace(
        create=create,
        wait_ready=AsyncMock(),
        retrieve_capabilities=AsyncMock(
            return_value=SimpleNamespace(
                capabilities=SimpleNamespace(telephony=False, proxy=True)
            )
        ),
    )

    class Client:
        def __init__(self, **kwargs):
            self.devices = devices

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setenv("MOBILERUN_CLOUD_DEVICE_TYPE", "ios_phone")
    from mobilerun_core_local.driver import cloud as driver

    client = Client()
    client.with_options = lambda **kwargs: client
    monkeypatch.setattr(driver, "AsyncMobilerun", lambda **kwargs: client)

    async def initialize():
        await cloud.CloudDevice(api_key="test", billing="paid")
        await cloud.CloudDevice(
            api_key="test", billing="paid", device_type="device_slot"
        )

    asyncio.run(initialize())
    assert [call.kwargs["device_type"] for call in create.await_args_list] == [
        "ios_phone",
        "device_slot",
    ]


@pytest.fixture
def fake_connection(monkeypatch):
    monkeypatch.setattr(AndroidDevice, "_ensure_connected", AsyncMock())


def run_device(device):
    async def ready():
        return await device

    return asyncio.run(ready())


def test_create_reuses_running_avd(fake_connection, monkeypatch):
    from mobilerun.clients.local import emulator

    monkeypatch.setenv("MOBILERUN_ANDROID_DEVICE_TYPE", "configured")
    monkeypatch.setattr(
        emulator, "find_running", AsyncMock(return_value="emulator-5554")
    )
    start = AsyncMock()
    monkeypatch.setattr(emulator, "start_avd", start)
    monkeypatch.setattr(emulator, "wait_boot", AsyncMock())
    device = run_device(AndroidDevice(avd="configured"))
    assert device._serial == "emulator-5554"
    emulator.find_running.assert_awaited_once_with("configured")
    start.assert_not_awaited()


def test_create_starts_explicit_avd_and_terminate(fake_connection, monkeypatch):
    from mobilerun.clients.local import emulator

    monkeypatch.setattr(emulator, "find_running", AsyncMock(return_value=None))
    monkeypatch.setattr(emulator, "list_avds", AsyncMock(return_value=["pixel"]))
    monkeypatch.setattr(emulator, "start_avd", AsyncMock(return_value="emulator-5556"))
    monkeypatch.setattr(emulator, "wait_boot", AsyncMock())
    monkeypatch.setattr(emulator, "stop", AsyncMock())
    device = run_device(AndroidDevice(avd="pixel", headless=True))
    assert device._serial == "emulator-5556"
    asyncio.run(device.terminate())
    emulator.stop.assert_awaited_once_with("emulator-5556")


def test_unknown_avd_lists_available_avds(fake_connection, monkeypatch):
    from mobilerun.clients.local import emulator

    monkeypatch.delenv("MOBILERUN_ANDROID_DEVICE_TYPE", raising=False)
    find_running = AsyncMock(return_value=None)
    monkeypatch.setattr(emulator, "find_running", find_running)
    monkeypatch.setattr(emulator, "list_avds", AsyncMock(return_value=["pixel"]))
    start_avd = AsyncMock()
    monkeypatch.setattr(emulator, "start_avd", start_avd)
    with pytest.raises(ValueError, match="pixel"):
        run_device(AndroidDevice(avd="missing"))
    find_running.assert_awaited_once_with("missing")
    start_avd.assert_not_awaited()


def test_terminate_refuses_physical_device(
    fake_connection,
):
    with pytest.raises(ValueError, match="emulator"):
        asyncio.run(AndroidDevice(serial="physical").terminate())


def test_client_lists_avds_and_finds_running(fake_connection, monkeypatch):
    from mobilerun.clients.local import emulator

    async def command(*args):
        if args[-1] == "-list-avds":
            return "pixel\ntablet\n"
        if args[-1] == "devices":
            return "List of devices attached\nemulator-5554\tdevice\nphysical\tdevice\n"
        return "pixel\nOK\n"

    monkeypatch.setattr(emulator, "_command", command)
    assert asyncio.run(emulator.list_avds()) == ["pixel", "tablet"]
    assert asyncio.run(emulator.find_running("pixel")) == "emulator-5554"
    assert asyncio.run(emulator.find_running("other")) is None


def test_client_start_uses_emulator_reported_port(fake_connection, monkeypatch):
    from mobilerun.clients.local import emulator

    async def launch(*args, **kwargs):
        assert "-port" not in args
        assert "-no-window" in args
        report = args[args.index("-report-console") + 1]
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", int(report.split(":")[1])
        )
        writer.write(b"5556")
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        return object()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    assert asyncio.run(emulator.start_avd("pixel", headless=True)) == "emulator-5556"


def test_wait_boot_timeout(fake_connection, monkeypatch):
    from mobilerun.clients.local import emulator

    async def command(*args):
        if args[-1] == "sys.boot_completed":
            await asyncio.sleep(1)
        return "0"

    monkeypatch.setattr(emulator, "_command", command)
    with pytest.raises(TimeoutError, match="did not boot"):
        asyncio.run(emulator.wait_boot("emulator-5554", timeout=0.01))


def test_find_running_uses_adb_property_when_console_unavailable(
    fake_connection, monkeypatch
):
    from mobilerun.clients.local import emulator

    async def command(*args):
        if args[-1] == "devices":
            return "List of devices attached\nemulator-5554\tdevice\n"
        if args[-1] == "name":
            return ""
        return "pixel\n"

    monkeypatch.setattr(emulator, "_command", command)
    assert asyncio.run(emulator.find_running("pixel")) == "emulator-5554"


def test_boot_failure_cleans_up_only_new_emulator(fake_connection, monkeypatch):
    from mobilerun.clients.local import emulator

    monkeypatch.setattr(emulator, "find_running", AsyncMock(return_value=None))
    monkeypatch.setattr(emulator, "list_avds", AsyncMock(return_value=["pixel"]))
    monkeypatch.setattr(emulator, "start_avd", AsyncMock(return_value="emulator-5556"))
    monkeypatch.setattr(
        emulator, "wait_boot", AsyncMock(side_effect=TimeoutError("boot"))
    )
    stop = AsyncMock()
    monkeypatch.setattr(emulator, "stop", stop)
    with pytest.raises(TimeoutError, match="boot"):
        run_device(AndroidDevice(avd="pixel"))
    stop.assert_awaited_once_with("emulator-5556")


def test_stop_kills_started_process_if_adb_unavailable(fake_connection, monkeypatch):
    from mobilerun.clients.local import emulator

    process = type(
        "Process",
        (),
        {"returncode": None, "kill": lambda self: None, "wait": AsyncMock()},
    )()
    monkeypatch.setitem(emulator._processes, "emulator-5556", process)
    monkeypatch.setattr(
        emulator, "_command", AsyncMock(side_effect=RuntimeError("offline"))
    )
    killed = []
    monkeypatch.setattr(process, "kill", lambda: killed.append(True))
    asyncio.run(emulator.stop("emulator-5556"))
    assert killed == [True]
