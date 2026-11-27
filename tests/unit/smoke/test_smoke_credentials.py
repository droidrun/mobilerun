"""Opt-in device credential, replay preflight, and cloud vault smoke tests."""

import asyncio
import json
import time
import traceback
from uuid import uuid4

import pytest

from mobilerun import AndroidDevice, RecordingDevice
from mobilerun.macro import MacroStep, build_macro
from mobilerun.macro.errors import MacroError
from mobilerun.macro.player import MacroPlayer
from tests.support.macro import read_run_steps
from tests.support.smoke_settings import CLOUD_ID, SERIAL

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
def test_record_secret_and_replay_wait(local_ready, tmp_path, caplog, monkeypatch):
    from mobilerun.core.credentials import DictCredentials
    from mobilerun.core.devices import AndroidDevice
    from mobilerun.core.devices.recording import RecordingDevice
    from mobilerun.macro.operator import MacroOperator

    caplog.set_level("DEBUG")
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(tmp_path / "macros"))

    async def run():
        device = AndroidDevice(serial=SERIAL, use_tcp=True, portal_mode="required")
        vault = await DictCredentials({"TEST": VALUE}).bind(device)
        await device.connect()
        recorder = RecordingDevice(device, runs_dir=tmp_path, capture="actions")
        try:
            await settings_search(device)
            await vault.type_field(None, "password", name="TEST")
            await recorder.wait(0.4)
            recorder.finish("succeeded")
            assert [row.action for row in read_run_steps(recorder.run_dir)] == [
                "extension_call",
                "wait",
            ]
            assert recorder.run_dir is not None
            steps = recorder.run_dir / "steps.jsonl"
            assert VALUE not in steps.read_text()
            assert VALUE not in build_macro(steps, "credential").read_text()
            # Avoid retyping the credential: select only the recorded wait.
            path = build_macro(steps, "wait", step_ids={2})
            started = time.monotonic()
            result = await MacroOperator(str(path), delay_between_actions=0).execute(
                device, ""
            )
            assert result.success and time.monotonic() - started >= 0.4
        finally:
            try:
                await device.press_button("home")
            finally:
                await recorder.close()

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


@pytest.mark.smoke_device
def test_protected_step_prevents_navigation(local_ready):
    async def run():
        async with RecordingDevice(AndroidDevice(SERIAL)) as recorder:
            macro = [
                MacroStep(i=1, action="button_press", args={"button": "home"}),
                MacroStep(
                    i=2,
                    action="type_secret",
                    args={"name": "example", "protected": True},
                ),
            ]
            with pytest.raises(MacroError, match="approval"):
                await MacroPlayer(recorder).replay_macro(macro)
            assert recorder.run_dir is None

    asyncio.run(run())


@pytest.mark.smoke_cloud
def test_missing_vault_field_preserves_http_status(request):
    request.getfixturevalue("real_env")
    from mobilerun.core.credentials import CloudCredentials
    from mobilerun.errors import CloudResourceNotFound

    async def run():
        vault = CloudCredentials(billing="no_commerce")
        value = uuid4().hex
        try:
            with pytest.raises(CloudResourceNotFound) as caught:
                await vault.update(
                    "ai.mobilerun.smoke.test", uuid4().hex, "password", value
                )
            assert caught.value.status == 404 and caught.value.value_safe
            assert value not in "".join(traceback.format_exception(caught.value))
        finally:
            await vault.close()

    asyncio.run(run())


@pytest.mark.smoke_cloud
def test_cloud_vault_write_and_cleanup(request, record_property):
    request.getfixturevalue("real_env")
    from mobilerun.core.credentials import CloudCredentials

    async def run():
        vault = CloudCredentials()
        package = "ai.mobilerun.smoke.test"
        name = f"smoke-{uuid4().hex}"
        record_property("created_vault_credential", f"{package}/{name}")
        try:
            await vault.save(package, name, {"password": "smoke-only-value"})
            items = await vault.list(package)
            assert any(
                item["name"] == name
                and item["package"] == package
                and set(item["fields"]) == {"password"}
                for item in items
            )
            assert any(item["name"] == name for item in await vault.list())
            await vault.save(
                package,
                name,
                {"password": "smoke-updated-value", "username": "smoke-user"},
            )
            await vault.update(package, name, "password", "smoke-final-value")
            item = await vault._find(package, name=name)
            assert item is not None
            assert set(item.fields) == {"password", "username"}
            assert item.fields["password"] == "smoke-final-value"
            await vault.delete(package, name, "username")
            assert any(
                item["name"] == name and set(item["fields"]) == {"password"}
                for item in await vault.list(package)
            )
        finally:
            try:
                await vault.delete(package, name)
                assert not any(
                    item["name"] == name for item in await vault.list(package)
                )
                record_property("deleted_vault_credential", f"{package}/{name}")
            finally:
                await vault.close()

    asyncio.run(run())
