"""Cloud authentication, device sessions, capability discovery and recovery."""

import asyncio
import json
import os
import stat
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import APIConnectionError, APIStatusError
from mobilerun_sdk.types.device_retrieve_capabilities_response import Capabilities

from mobilerun.clients.cloud import api_keys, apps, devices, proxies, tasks
from mobilerun.clients.cloud import device_flow as deviceauth
from mobilerun.clients.local import login_store as credentials
from mobilerun.core.devices.components import cloud_lifecycle as lifecycle
from mobilerun.errors import CloudValidationError, DeviceOperationNotSupported
from mobilerun.errors.cloud import (
    CloudAuthenticationError,
    CloudConnectionError,
    CloudPermissionDenied,
    CloudServiceUnavailable,
)
from mobilerun.models.apps import App
from mobilerun.models.billing import Billing


def fake_client(monkeypatch, error=None):
    endpoint = AsyncMock(side_effect=error)
    context = AsyncMock()
    context.__aenter__.return_value = SimpleNamespace(
        devices=SimpleNamespace(list=endpoint)
    )

    def factory(**kwargs):
        return context

    monkeypatch.setattr(api_keys, "cloud_client", factory)
    return endpoint


def test_valid_key_uses_sdk(monkeypatch):
    endpoint = fake_client(monkeypatch)
    asyncio.run(api_keys.validate_api_key("example-key", billing="included"))
    assert endpoint.await_args.kwargs["page_size"] == 1


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, CloudAuthenticationError),
        (403, CloudPermissionDenied),
        (500, CloudServiceUnavailable),
    ],
)
def test_http_errors(monkeypatch, status, expected):
    response = httpx.Response(
        status, request=httpx.Request("GET", "https://cloud.example/devices")
    )
    error = APIStatusError("private response", response=response, body=None)
    fake_client(monkeypatch, error)
    with pytest.raises(expected, match=f"HTTP {status}") as raised:
        asyncio.run(api_keys.validate_api_key("key", billing="included"))
    assert raised.value.__cause__ is error
    assert "private response" not in str(raised.value)


def test_network_error(monkeypatch):
    error = APIConnectionError(request=httpx.Request("GET", "https://example.com"))
    fake_client(monkeypatch, error)
    with pytest.raises(CloudConnectionError):
        asyncio.run(api_keys.validate_api_key("key", billing="included"))


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_save_token_writes_private_file_and_repairs_directory_permissions(
    monkeypatch, tmp_path
):
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir(mode=0o755)
    credential_file = credential_dir / "mobilerun-cloud.json"
    monkeypatch.setattr(credentials, "credential_path", lambda: credential_file)

    path = credentials.save_token("secret-token", "https://cloud.mobilerun.ai/api/auth")

    assert path == credential_file
    assert json.loads(path.read_text()) == {
        "access_token": "secret-token",
        "auth_url": "https://cloud.mobilerun.ai/api/auth",
    }
    if os.name != "nt":
        assert _mode(credential_dir) == 0o700
        assert _mode(path) == 0o600


def test_save_token_overwrites_existing_token_privately(monkeypatch, tmp_path):
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir()
    credential_file = credential_dir / "mobilerun-cloud.json"
    credential_file.write_text("old")
    if os.name != "nt":
        os.chmod(credential_file, 0o644)
    monkeypatch.setattr(credentials, "credential_path", lambda: credential_file)

    credentials.save_token("new-token", "https://auth.example")

    assert json.loads(credential_file.read_text()) == {
        "access_token": "new-token",
        "auth_url": "https://auth.example",
    }
    if os.name != "nt":
        assert _mode(credential_file) == 0o600


def test_clear_credentials_removes_file_and_is_idempotent(monkeypatch, tmp_path):
    credential_file = tmp_path / "credentials" / "mobilerun-cloud.json"
    credential_file.parent.mkdir()
    credential_file.write_text("{}")
    monkeypatch.setattr(credentials, "credential_path", lambda: credential_file)

    assert credentials.clear_credentials() is True
    assert credential_file.exists() is False
    assert credentials.clear_credentials() is False


def _device_code(*, expires_in=60):
    return deviceauth.CodeResponse(
        device_code="device-code",
        user_code="USER",
        verification_uri="https://example.test",
        verification_uri_complete="https://example.test/complete",
        expires_in=expires_in,
        interval=1,
    )


@pytest.mark.parametrize(
    "reason,second_delay", [("authorization_pending", 2.0), ("slow_down", 5.0)]
)
def test_poll_token_retries_until_approved(monkeypatch, reason, second_delay):
    monkeypatch.setenv("MOBILERUN_DEVICE_AUTH_MIN_POLL_INTERVAL", "2.0")
    monkeypatch.setenv("MOBILERUN_DEVICE_AUTH_SLOW_DOWN_BUMP", "3.0")
    code = _device_code()
    calls = []
    sleeps = []

    async def fake_exchange(_auth_url, _client_id, _device_code):
        calls.append("exchange")
        if len(calls) == 1:
            raise deviceauth.DeviceAuthError(reason)
        return "token"

    monkeypatch.setattr(deviceauth, "_exchange", fake_exchange)
    monkeypatch.setattr(
        deviceauth.asyncio, "sleep", AsyncMock(side_effect=sleeps.append)
    )
    monkeypatch.setattr(deviceauth.time, "monotonic", lambda: len(calls))

    assert (
        asyncio.run(
            deviceauth.poll_token(
                "https://auth.example", "client", code, billing="included"
            )
        )
        == "token"
    )
    assert calls == ["exchange", "exchange"]
    assert sleeps == [2.0, second_delay]


def test_poll_token_expires_pending_authorization(monkeypatch):
    code = _device_code(expires_in=0)

    monkeypatch.setattr(
        deviceauth,
        "_exchange",
        AsyncMock(side_effect=deviceauth.DeviceAuthError("authorization_pending")),
    )
    monkeypatch.setattr(deviceauth.asyncio, "sleep", AsyncMock())
    ticks = iter([0, 1])
    monkeypatch.setattr(
        deviceauth,
        "time",
        type("Clock", (), {"monotonic": staticmethod(lambda: next(ticks))}),
    )

    with pytest.raises(deviceauth.DeviceAuthError, match="expired_token"):
        asyncio.run(
            deviceauth.poll_token(
                "https://auth.example", "client", code, billing="included"
            )
        )


def test_poll_token_raises_terminal_denial(monkeypatch):
    code = _device_code()

    monkeypatch.setattr(
        deviceauth,
        "_exchange",
        AsyncMock(side_effect=deviceauth.DeviceAuthError("access_denied", "denied")),
    )
    monkeypatch.setattr(deviceauth.asyncio, "sleep", AsyncMock())

    with pytest.raises(deviceauth.DeviceAuthError, match="access_denied"):
        asyncio.run(
            deviceauth.poll_token(
                "https://auth.example", "client", code, billing="included"
            )
        )


def _device(monkeypatch, *, api_key=None, base_url=None):
    cloud = pytest.importorskip("mobilerun_core_local.driver.cloud")
    from mobilerun.core.devices.cloud import CloudDevice

    calls = []

    def init(self, device_id, api_key=None, base_url=None, **kwargs):
        calls.append((device_id, api_key, base_url, kwargs))
        self.device_id = device_id
        self._client = None
        self._write_client = None

    monkeypatch.setattr(cloud.CloudDriver, "__init__", init)
    device = CloudDevice("device-1", api_key=api_key, base_url=base_url)
    return device, calls[0]


@pytest.mark.parametrize(
    "explicit,expected",
    [("explicit-secret", "explicit-secret"), (None, "environment-secret")],
    ids=["explicit", "environment"],
)
def test_key_precedence_over_saved_login(monkeypatch, tmp_path, explicit, expected):
    monkeypatch.setenv("MOBILERUN_API_KEY", "environment-secret")
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))
    path = tmp_path / "credentials" / "mobilerun-cloud.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"api_key": "saved-secret"}))

    _, call = _device(monkeypatch, api_key=explicit)
    assert call[1] == expected


@pytest.mark.parametrize("field", ["api_key", "access_token"])
def test_saved_login_key_variants(monkeypatch, tmp_path, field):
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))
    path = tmp_path / "credentials" / "mobilerun-cloud.json"
    path.parent.mkdir()
    path.write_text(json.dumps({field: "saved-secret"}))

    _, call = _device(monkeypatch)

    assert call[1] == "saved-secret"


def test_missing_credentials_raises_safe_auth_error(monkeypatch, tmp_path):
    pytest.importorskip("mobilerun_core_local.driver.cloud")
    from mobilerun.core.devices.cloud import CloudDevice
    from mobilerun.errors import CloudAuthRequired

    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))

    with pytest.raises(CloudAuthRequired) as error:
        CloudDevice("device-1")
    assert "mobilerun account login" in str(error.value)
    assert "secret" not in str(error.value)


def test_default_base_url_comes_from_settings(monkeypatch):
    from mobilerun.config.settings import get_settings

    monkeypatch.setenv("MOBILERUN_API_KEY", "test-secret")
    _, call = _device(monkeypatch)

    assert call[2] == get_settings().MOBILERUN_CLOUD_BASE_URL


def test_credentials_are_not_exposed_in_device_repr_or_string(monkeypatch):
    monkeypatch.setenv("MOBILERUN_API_KEY", "never-print-this")
    device, _ = _device(monkeypatch)

    assert "never-print-this" not in repr(device)
    assert "never-print-this" not in str(device)


@pytest.mark.parametrize(
    "existing,keep,fail",
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (False, False, True),
    ],
)
def test_cloud_close_closes_owned_client(monkeypatch, existing, keep, fail):
    from mobilerun_core_local.driver import cloud

    from mobilerun.core.devices import CloudDevice

    client = SimpleNamespace(close=AsyncMock())
    client.with_options = lambda **kwargs: client
    monkeypatch.setattr(cloud, "AsyncMobilerun", lambda **kwargs: client)
    terminate = AsyncMock(
        side_effect=RuntimeError("terminate failed") if fail else None
    )
    monkeypatch.setattr(lifecycle.devices, "terminate", terminate)
    device = CloudDevice("existing" if existing else None, api_key="test", keep=keep)
    if not existing:
        device.device_id = "created"
        device._owned = True

    async def run():
        if fail:
            with pytest.raises(RuntimeError, match="terminate failed"):
                await device.close()
            client.close.assert_not_awaited()
        else:
            await device.close()
            client.close.assert_awaited_once()

    asyncio.run(run())


def test_failed_termination_can_retry_with_same_owned_sdk_client(monkeypatch):
    import httpx
    from mobilerun_core_local.driver import cloud
    from mobilerun_sdk import AsyncMobilerun

    from mobilerun.core.devices import CloudDevice

    attempts = []

    def respond(request):
        attempts.append(request)
        return (
            httpx.Response(400, json={"error": "failed"})
            if len(attempts) == 1
            else httpx.Response(204)
        )

    sdk = AsyncMobilerun(
        api_key="test",
        base_url="https://example.test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setattr(cloud, "AsyncMobilerun", lambda **kwargs: sdk)
    device = CloudDevice(api_key="test")
    device.device_id = "created"
    device._owned = True

    async def run():
        with pytest.raises(CloudValidationError):
            await device.close()
        await device.close()
        assert len(attempts) == 2
        assert sdk.is_closed()

    asyncio.run(run())


def test_capabilities_uses_given_client():
    retrieve = AsyncMock(
        return_value=SimpleNamespace(capabilities=SimpleNamespace(telephony=True))
    )
    result = asyncio.run(
        devices.capabilities(
            "phone",
            client=SimpleNamespace(
                devices=SimpleNamespace(retrieve_capabilities=retrieve)
            ),
            billing=Billing.NO_COMMERCE,
        )
    )
    assert result.capabilities.telephony
    retrieve.assert_awaited_once_with("phone")


def test_capabilities_opens_client_when_needed(monkeypatch):
    retrieve = AsyncMock(return_value="capabilities")
    context = AsyncMock()
    context.__aenter__.return_value = SimpleNamespace(
        devices=SimpleNamespace(retrieve_capabilities=retrieve)
    )

    def factory(*args, **kwargs):
        return context

    monkeypatch.setattr(devices, "cloud_client", factory)
    assert (
        asyncio.run(
            devices.capabilities(
                "phone",
                billing=Billing.NO_COMMERCE,
                api_key="fake",
                base_url="https://example.test",
            )
        )
        == "capabilities"
    )
    retrieve.assert_awaited_once_with("phone")
    context.__aexit__.assert_awaited_once()


def test_capabilities_maps_sdk_error():
    error = APIStatusError(
        "private response",
        response=httpx.Response(
            403, request=httpx.Request("GET", "https://example.test")
        ),
        body={},
    )
    retrieve = AsyncMock(side_effect=error)
    with pytest.raises(CloudPermissionDenied) as raised:
        asyncio.run(
            devices.capabilities(
                "phone",
                billing=Billing.NO_COMMERCE,
                client=SimpleNamespace(
                    devices=SimpleNamespace(retrieve_capabilities=retrieve)
                ),
            )
        )
    assert raised.value.__cause__ is error
    assert "private response" not in str(raised.value)


@pytest.mark.parametrize("supported", [False, True, None])
def test_proxy_capability_controls_adoption_and_operations(monkeypatch, supported):
    from mobilerun.core.devices import CloudDevice

    phone = CloudDevice("phone", api_key="test")
    info = SimpleNamespace(
        capabilities=SimpleNamespace(telephony=False, proxy=supported)
    )
    monkeypatch.setattr(devices, "capabilities", AsyncMock(return_value=info))
    status = AsyncMock(
        return_value=SimpleNamespace(connected=False, name=None, protocol=None)
    )
    monkeypatch.setattr(proxies, "device_status", status)
    asyncio.run(phone.connect())
    assert status.await_count == int(supported is not False)
    assert ("proxy_switch" in phone.supported_operations) == (supported is not False)


@pytest.mark.parametrize("code", ["DEVICE_TOOL_UNSUPPORTED", "OTHER"])
def test_missing_proxy_capability_only_tolerates_unsupported(monkeypatch, code):
    from mobilerun.core.devices import CloudDevice

    phone = CloudDevice("phone", api_key="test")
    info = SimpleNamespace(capabilities=Capabilities.construct(telephony=False))
    monkeypatch.setattr(devices, "capabilities", AsyncMock(return_value=info))
    sdk_error = APIStatusError(
        "invalid request",
        response=httpx.Response(
            400, request=httpx.Request("GET", "https://example.test")
        ),
        body={"code": code},
    )
    status = AsyncMock(side_effect=sdk_error)
    client = SimpleNamespace(
        devices=SimpleNamespace(proxy=SimpleNamespace(status=status))
    )
    monkeypatch.setattr(proxies, "cloud_client", lambda *a, **kw: nullcontext(client))
    if code == "DEVICE_TOOL_UNSUPPORTED":
        asyncio.run(phone.connect())
        assert "proxy_switch" not in phone.supported_operations
    else:
        with pytest.raises(CloudValidationError):
            asyncio.run(phone.connect())
    status.assert_awaited_once()


def test_unknown_proxy_handles_typed_error_without_cause(monkeypatch):
    from mobilerun.core.devices import CloudDevice

    phone = CloudDevice("phone", api_key="test")
    info = SimpleNamespace(capabilities=SimpleNamespace(telephony=False, proxy=None))
    monkeypatch.setattr(devices, "capabilities", AsyncMock(return_value=info))
    monkeypatch.setattr(
        proxies,
        "device_status",
        AsyncMock(side_effect=DeviceOperationNotSupported("Unsupported proxy status")),
    )
    asyncio.run(phone.connect())
    assert "proxy_switch" not in phone.supported_operations


def test_install_polls_background_request(monkeypatch):
    from mobilerun.core.devices import CloudDevice

    phone = CloudDevice("phone", api_key="test")
    app = App("pkg", "1", "Example", id="lib")

    async def submit(*args, background=False, **kwargs):
        if not background:
            raise TimeoutError("foreground install exceeded SDK timeout")

    monkeypatch.setattr(apps, "install", submit)
    attempt = SimpleNamespace(app_id="pkg", started_at=1, status="succeeded")
    polls = AsyncMock(
        side_effect=[SimpleNamespace(installs=[]), SimpleNamespace(installs=[attempt])]
    )
    monkeypatch.setattr(apps, "list_installs", polls)
    monkeypatch.setattr(
        phone,
        "list_apps",
        AsyncMock(return_value=[{"package_name": "pkg", "version_name": "1"}]),
    )
    assert asyncio.run(phone.install_app(app)) == app
    assert polls.await_count == 2


def test_progress_disconnect_polls_without_stopping(monkeypatch):
    monkeypatch.setattr(tasks, "run", AsyncMock(return_value=SimpleNamespace(id="id")))
    monkeypatch.setattr(
        tasks, "_progress", AsyncMock(side_effect=httpx.ReadError("disconnect"))
    )
    status = AsyncMock(
        return_value=SimpleNamespace(
            status="completed", succeeded=True, message="ok", steps=1
        )
    )
    monkeypatch.setattr(tasks, "status", status)
    stop = AsyncMock()
    monkeypatch.setattr(tasks, "_stop", stop)
    result = asyncio.run(
        tasks.run_task(None, device_id="phone", task="fake", billing=Billing.PAID)
    )
    assert result.success
    status.assert_awaited_once()
    stop.assert_not_awaited()
