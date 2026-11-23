"""Credential providers and secret-safe device/agent integration, entirely offline."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.credentials import (
    CloudCredentials,
    DictCredentials,
    EnvCredentials,
    FileCredentials,
)
from mobilerun.core.devices.base import Device as DeviceMixin
from mobilerun.core.devices.pool import DevicePool
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.devices.stealth import StealthDevice
from mobilerun.macro.player import MacroPlayer


class FakeDevice(DeviceMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_text = AsyncMock(return_value=True)
        self._input_text_raw = self.input_text
        self.state.refresh = AsyncMock(
            return_value=SimpleNamespace(
                phone_state={
                    "focusedElement": {"className": "EditText"},
                    "isEditable": True,
                }
            )
        )


def test_dict_normalizes_flat_and_nested_and_redacts():
    provider = DictCredentials(
        {
            "gmail_password": "private-value",
            "com.gmail": {
                "work": {"email": "private-email", "password": "private-password"}
            },
        }
    )
    asyncio.run(provider.bind(FakeDevice()))
    items = asyncio.run(provider.list())
    assert [(item["package"], item["name"], item["fields"]) for item in items] == [
        (None, "gmail_password", ["password"]),
        ("com.gmail", "work", ["email", "password"]),
    ]
    assert "private-" not in repr(items)
    assert "private-" not in str(items[0])
    device = provider.device
    asyncio.run(provider.type_field("com.gmail", "email", "work"))
    device.input_text.assert_awaited_once_with("private-email")


@pytest.mark.parametrize("extension", ["yaml", "json"])
def test_file_normalizes_flat_and_nested(tmp_path, extension):
    path = tmp_path / f"credentials.{extension}"
    path.write_text(
        json.dumps(
            {
                "gmail_password": "private-value",
                "com.gmail": {"work": {"username": "private-name"}},
            }
        )
    )
    path.chmod(0o600)
    provider = FileCredentials(path)
    device = FakeDevice()
    asyncio.run(provider.bind(device))
    asyncio.run(provider.type_field(None, "password", "gmail_password"))
    asyncio.run(provider.type_field("com.gmail", "username", "work"))
    assert [call.args[0] for call in device.input_text.await_args_list] == [
        "private-value",
        "private-name",
    ]


def test_yaml_flat(tmp_path):
    path = tmp_path / "credentials.yaml"
    path.write_text("gmail_password: private-value\n")
    path.chmod(0o600)
    provider = FileCredentials(path)
    asyncio.run(provider.bind(FakeDevice()))
    assert asyncio.run(provider.list())[0]["name"] == "gmail_password"


def test_env_normalizes_flat(monkeypatch):
    monkeypatch.setenv("MOBILERUN_SECRET_GMAIL_PASSWORD", "private-value")
    provider = EnvCredentials()
    device = FakeDevice()
    asyncio.run(provider.bind(device))
    asyncio.run(provider.type_field(None, "password", "GMAIL_PASSWORD"))
    device.input_text.assert_awaited_once_with("private-value")


def test_cloud_paginated_cached_and_login_key(monkeypatch):

    calls = []

    async def list_page(*, page):
        calls.append(page)
        return SimpleNamespace(
            items=[
                SimpleNamespace(
                    package_name=f"com.app{page}",
                    credential_name="work",
                    fields=[
                        SimpleNamespace(field_type="password", value="private-value")
                    ],
                    source="mobilerun" if page == 1 else "onepassword",
                )
            ],
            pagination=SimpleNamespace(has_next=page < 2),
        )

    async def list_package(package):
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    package_name=package,
                    credential_name="work",
                    fields=[
                        SimpleNamespace(field_type="password", value="private-value")
                    ],
                )
            ]
        )

    client = SimpleNamespace(
        credentials=SimpleNamespace(
            list=list_page, packages=SimpleNamespace(list=list_package)
        ),
        close=AsyncMock(),
    )
    constructor = []

    def make_client(**kwargs):
        constructor.append(kwargs)
        return client

    from mobilerun.clients.cloud import client as transport

    monkeypatch.setattr(transport, "AsyncMobilerun", make_client)
    monkeypatch.setattr(transport, "resolve_cloud_api_key", lambda: "fake-login-key")
    provider = CloudCredentials(base_url="https://fake.invalid")

    async def run():
        await provider.bind(FakeDevice())
        assert len(await provider.list()) == 2
        assert (await provider.list("com.app2"))[0]["name"] == "work"
        assert "private-value" not in repr(await provider.list())
        await provider.close()

    asyncio.run(run())
    assert calls == [1, 2]
    assert constructor == [
        {"api_key": "fake-login-key", "base_url": "https://fake.invalid"}
    ]


def test_device_order_recording_wrappers_pool_and_no_log(caplog):
    caplog.set_level("DEBUG")
    first = DictCredentials({"login": "private-first"})
    second = DictCredentials(
        {"login": "private-second", "other": "private-other"}, name="second"
    )
    device = FakeDevice()
    asyncio.run(first.bind(device))
    asyncio.run(second.bind(device))
    recording = RecordingDevice(device)
    stealth = StealthDevice(recording)
    assert dict(recording.extensions) == {"vault": first, "second": second}
    assert dict(stealth.extensions) == dict(recording.extensions)

    async def run():
        assert await first.type_field(None, "password", name="login") is None
        assert await second.type_field(None, "password", name="other") is None
        assert DevicePool([recording]).extensions[0]["vault"] is first
        assert await first.type_field(None, "password", name="login") is None

    asyncio.run(run())
    assert [call.args[0] for call in device.input_text.await_args_list] == [
        "private-first",
        "private-other",
        "private-first",
    ]
    assert recording.recorded_actions[0]["name"] == "vault.type_field"
    assert "private-" not in repr(recording.recorded_actions)
    assert "private-" not in caplog.text


@pytest.mark.parametrize(
    "reference",
    [
        {"package": "com.gmail", "field": "email", "name": "work"},
        {"secret": "login"},
        {"secret_id": "login"},
    ],
)
def test_macro_uses_device_credentials(reference, caplog):
    caplog.set_level("DEBUG")
    device = FakeDevice()
    provider = DictCredentials(
        {
            "login": "private-value",
            "com.gmail": {"work": {"email": "private-value"}},
        }
    )
    asyncio.run(provider.bind(device))
    assert asyncio.run(
        MacroPlayer(device).replay_action({"action_type": "type_secret", **reference})
    )
    device.input_text.assert_awaited_once_with("private-value")
    assert "private-value" not in caplog.text
