import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner
from mobilerun_core_local.driver.android.portal import (
    PORTAL_PACKAGE_NAME,
    portal_a11y_service,
)

from mobilerun.cli.devices import device as cli
from mobilerun.clients.portal.state_fetch import fetch_state_with_retry
from mobilerun.core.devices import AndroidDevice, DevicePool, android
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.operator import Operator
from mobilerun.errors import StateFetchFailed
from mobilerun.models.results import ExecutionResult
from tests.support.devices import MemoryDevice


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
