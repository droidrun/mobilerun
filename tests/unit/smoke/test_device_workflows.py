"""Opt-in device, CLI, Counter installation, and recording workflows."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.macro import read_run_steps
from tests.support.smoke_settings import CLOUD_ID, SERIAL


async def exercise(device):
    await device.connect()
    try:
        assert (await device.screenshot()).startswith(b"\x89PNG\r\n\x1a\n")
        await device.start_app("com.android.settings")
        state = await device.state.refresh()
        assert state.elements
        assert await device.list_apps()
        await device.swipe(200, 800, 200, 400, duration_ms=300)
        await device.tap(100, 100)
    finally:
        await device.press_button("home")


@pytest.mark.smoke_device
def test_local_device(local_ready):
    from mobilerun.core.devices import AndroidDevice

    asyncio.run(exercise(AndroidDevice(serial=SERIAL)))


@pytest.mark.smoke_device
def test_local_wrapper_and_pool_state(local_ready):
    from mobilerun import AndroidDevice, DevicePool, RecordingDevice

    async def run():
        async with AndroidDevice(SERIAL) as root:
            wrapped = RecordingDevice(root)
            await wrapped.connect()
            await wrapped.connect()
            try:
                await wrapped.start_app("com.android.settings")
                states = await DevicePool([wrapped]).state.refresh()
                assert states[0] is wrapped.state.current
                assert states[0].elements
                await wrapped.press_button("home")
                row = read_run_steps(wrapped.run_dir)[-1]
                assert row.ok and row.action == "button_press"
                assert row.args == {"button": "home"}
                wrapped.finish("succeeded")
            finally:
                await root.press_button("home")

    asyncio.run(run())


@pytest.mark.smoke_cloud
def test_cloud_device(cloud_ready):
    from mobilerun.core.devices.cloud import CloudDevice

    asyncio.run(exercise(CloudDevice(device_id=CLOUD_ID)))


@pytest.mark.smoke_cloud
def test_cloud_pool(cloud_ready):
    from mobilerun.core.devices.cloud import CloudDevice
    from mobilerun.core.devices.pool import DevicePool

    async def run():
        device = CloudDevice(device_id=CLOUD_ID)
        await device.connect()
        try:
            images = await DevicePool([device]).screenshot()
            assert len(images) == 1 and images[0].startswith(b"\x89PNG\r\n\x1a\n")
        finally:
            await device.press_button("home")

    asyncio.run(run())


def cli(*args, env):
    return subprocess.run(
        [str(Path(sys.executable).with_name("mobilerun")), *args],
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
    )


@pytest.mark.smoke_device
def test_single_adb_device_default(local_ready):
    result = subprocess.run(
        ["adb", "devices"], capture_output=True, text=True, check=True
    )
    devices = [
        line.split()[0]
        for line in result.stdout.splitlines()[1:]
        if line.endswith("\tdevice")
    ]
    if devices != [SERIAL]:
        pytest.skip("requires exactly the configured local adb device")
    env = os.environ.copy()
    # This case exercises local-only discovery, not mixed local/cloud selection.
    env.pop("MOBILERUN_API_KEY", None)
    result = cli("device", "app", "list", env=env)
    assert result.returncode == 0, result.stderr
    assert f"Using {SERIAL}" in result.stderr
    assert result.stdout.strip()


@pytest.mark.smoke_device
def test_config_default(local_ready, tmp_path):
    from mobilerun.config.migrations import CURRENT_VERSION

    config = tmp_path / "device.yaml"
    config.write_text(f"_version: {CURRENT_VERSION}\ndevice:\n  serial: {SERIAL}\n")
    env = os.environ.copy()
    result = cli("device", "app", "list", "--config", str(config), env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


@pytest.mark.smoke_cloud
def test_cloud_config_default(cloud_ready, tmp_path):
    from mobilerun.config.migrations import CURRENT_VERSION

    config = tmp_path / "device.yaml"
    config.write_text(f"_version: {CURRENT_VERSION}\ndevice:\n  serial: {CLOUD_ID}\n")
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_path)
    result = cli("device", "screen", "screenshot", "--config", str(config), env=env)
    assert result.returncode == 0, result.stderr
    image = Path(result.stdout.strip())
    assert image.is_relative_to(tmp_path)
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.fixture
def counter_apk(real_smoke_apk):
    from mobilerun.clients.local.apk import inspect_apk

    path = real_smoke_apk
    if not path.is_file():
        pytest.skip(f"Counter smoke APK missing: {path}; set MOBILERUN_SMOKE_APK")
    package, _, _, _ = inspect_apk(path)
    if package != "ai.droidrun.counter":
        pytest.skip("Smoke APK is not the Counter app; refusing to modify another app")
    return path


def _has_counter(apps):
    return any(
        (app.get("package") or app.get("package_name") or app.get("packageName"))
        == "ai.droidrun.counter"
        for app in apps
    )


@pytest.mark.smoke_device
def test_local_counter_install(local_ready, counter_apk):
    from mobilerun.core.devices import AndroidDevice

    async def run():
        device = AndroidDevice(serial=SERIAL)
        await device.connect()
        app = await device.install_apk(str(counter_apk), reinstall=True)
        assert app.package == "ai.droidrun.counter"
        try:
            await device.start_app(app.package)
            assert _has_counter(await device.list_apps())
            await device.uninstall_app(app.package)
        finally:
            await device.install_apk(str(counter_apk), reinstall=True)
        assert _has_counter(await device.list_apps())

    asyncio.run(run())


@pytest.mark.smoke_cloud
def test_cloud_counter_upload_install_and_uninstall(cloud_ready, counter_apk):
    from mobilerun.core.devices.cloud import CloudDevice

    async def run():
        device = CloudDevice(device_id=CLOUD_ID)
        await device.connect()
        app = await device.install_apk(counter_apk)
        assert app.package == "ai.droidrun.counter"
        try:
            await device.start_app(app.package)
            assert _has_counter(await device.list_apps())
            await device.uninstall_app(app.package)
        finally:
            await device.install_app(app)
        assert _has_counter(await device.list_apps())

    asyncio.run(run())


@pytest.mark.smoke_device
def test_record_named_home_then_replay_wait(local_ready, tmp_path, monkeypatch):
    from mobilerun import AndroidDevice, RecordingDevice
    from mobilerun.core.devices import Device
    from mobilerun.core.extensions import Extension, device_action
    from mobilerun.macro import MacroPlayer, build_macro, load_macro

    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(tmp_path / "macros"))

    class Navigation(Extension):
        @device_action()
        async def home(self, device: Device) -> None:
            """Return the emulator to its home screen."""
            await device.press_button("home")

    async def run():
        async with AndroidDevice(SERIAL) as device:
            recorder = RecordingDevice(device, runs_dir=tmp_path, capture="actions")
            navigation = await Navigation(name="navigation").bind(recorder)
            await navigation.home()
            await recorder.wait(0.01)
            recorder.finish("succeeded")
            home, wait = read_run_steps(recorder.run_dir)
            assert home.ok and home.action == "extension_call"
            assert home.args == {"name": "navigation.home", "arguments": {}}
            assert wait.ok and wait.action == "wait" and wait.args == {"duration": 0.01}
            assert recorder.run_dir is not None
            path = build_macro(recorder.run_dir / "steps.jsonl", "home", step_ids={2})
            assert await MacroPlayer(device).replay_macro(load_macro(path))

    asyncio.run(run())
