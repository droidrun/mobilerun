"""Exercise device credentials and recorded waits without real secrets."""

import asyncio
import json
import time

import pytest
from conftest import CLOUD_ID, SERIAL

VALUE = "demo-user"


async def settings_search(device):
    await device.press_button("back")
    await device.start_app("com.android.settings")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state = await device.state.refresh()
        field = next(
            (
                element
                for element in state.elements
                if element.get("className") == "EditText"
            ),
            None,
        )
        if field:
            await device.tap_element(field["index"])
            await asyncio.sleep(0.5)
            return
        search = next(
            (
                element
                for element in state.elements
                if element.get("resourceId", "").endswith("search_action_bar")
            ),
            None,
        )
        if search:
            await device.tap_element(search["index"])
        await asyncio.sleep(0.5)
    raise AssertionError(
        "Settings search input did not become available within 15 seconds"
    )


@pytest.mark.smoke_device
@pytest.mark.parametrize("kind", ["dict", "file", "env"])
def test_credentials_type_real_field(local_ready, tmp_path, monkeypatch, caplog, kind):
    from mobilerun.core.credentials import (
        DictCredentials,
        EnvCredentials,
        FileCredentials,
    )
    from mobilerun.core.devices import AndroidDevice

    if kind == "dict":
        provider = DictCredentials({"TEST": VALUE})
    elif kind == "file":
        path = tmp_path / "credentials.json"
        path.write_text(json.dumps({"TEST": VALUE}))
        provider = FileCredentials(path)
    else:
        monkeypatch.setenv("MOBILERUN_SECRET_TEST", VALUE)
        provider = EnvCredentials()
    caplog.set_level("DEBUG")

    async def run():
        device = AndroidDevice(serial=SERIAL, use_tcp=True, portal_mode="required")
        await provider.bind(device)
        await device.connect()
        try:
            await settings_search(device)
            await provider.type_field(None, "password", name="TEST")
            state = await device.state.refresh()
            assert VALUE in repr(state.elements)
        finally:
            await device.press_button("home")

    asyncio.run(run())
    assert VALUE not in caplog.text


@pytest.mark.smoke_device
def test_record_secret_and_replay_wait(local_ready, tmp_path, caplog):
    from mobilerun.core.credentials import DictCredentials
    from mobilerun.core.devices import AndroidDevice
    from mobilerun.core.devices.recording import RecordingDevice
    from mobilerun.macro.operator import MacroOperator

    caplog.set_level("DEBUG")

    async def run():
        device = AndroidDevice(serial=SERIAL, use_tcp=True, portal_mode="required")
        vault = await DictCredentials({"TEST": VALUE}).bind(device)
        await device.connect()
        recorder = RecordingDevice(device)
        try:
            await settings_search(device)
            await vault.type_field(None, "password", name="TEST")
            await recorder.wait(0.4)
            assert [a["action_type"] for a in recorder.recorded_actions] == [
                "extension_call",
                "wait",
            ]
            assert VALUE not in repr(recorder.recorded_actions)
            path = tmp_path / "macro.json"
            path.write_text(
                json.dumps(
                    {
                        "macro_schema_version": "2.0",
                        "actions": recorder.recorded_actions,
                    }
                )
            )
            assert VALUE not in path.read_text()
            # Extension-call replay is not implemented; replay the recorded wait.
            path.write_text(
                json.dumps(
                    {
                        "macro_schema_version": "2.0",
                        "actions": [recorder.recorded_actions[-1]],
                    }
                )
            )
            started = time.monotonic()
            result = await MacroOperator(str(path), delay_between_actions=0).execute(
                device, ""
            )
            assert result.success and time.monotonic() - started >= 0.4
        finally:
            await device.press_button("home")

    asyncio.run(run())
    assert VALUE not in caplog.text


@pytest.mark.smoke_cloud
def test_cloud_wait(cloud_ready):
    from mobilerun.core.devices.cloud import CloudDevice

    async def run():
        device = CloudDevice(device_id=CLOUD_ID)
        await device.connect()
        started = time.monotonic()
        await device.wait(0.25)
        assert time.monotonic() - started >= 0.25
        await device.press_button("home")

    asyncio.run(run())


@pytest.mark.smoke_cloud
def test_cloud_credentials_list(cloud_ready):
    from mobilerun.core.credentials import CloudCredentials

    async def run():
        provider = CloudCredentials()
        try:
            items = await provider.list()
            if not items:
                pytest.skip("Cloud vault is empty")
            assert all(
                item["name"]
                and (item["package"] is None or isinstance(item["package"], str))
                for item in items
            )
        finally:
            await provider.close()

    asyncio.run(run())
