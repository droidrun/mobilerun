"""Independent optional resources bind; proxy/operator behavior stays on phones."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.credentials import (
    CloudCredentials,
    DictCredentials,
    EnvCredentials,
    FileCredentials,
)
from mobilerun.core.devices import (
    AndroidDevice,
    CloudDevice,
    IOSDevice,
    RecordingDevice,
    StealthDevice,
)
from mobilerun.core.mailbox import CloudMailbox
from mobilerun.core.proxy import CloudRotatingProxy
from mobilerun.errors import DeviceOperationNotSupported, ProxyNotActive


@pytest.mark.parametrize("kind", ["dict", "file", "env", "cloud"])
def test_every_provider_binds_and_unbinds_in_order(kind, tmp_path, monkeypatch):
    path = tmp_path / "credentials.yaml"
    path.write_text("login: example-only\n")
    path.chmod(0o600)
    monkeypatch.setenv("MOBILERUN_SECRET_LOGIN", "example-only")
    provider = {
        "dict": lambda: DictCredentials({}, name="second"),
        "file": lambda: FileCredentials(path, name="second"),
        "env": lambda: EnvCredentials(name="second"),
        "cloud": lambda: CloudCredentials(api_key="fake", name="second"),
    }[kind]()
    phone = AndroidDevice("local")
    first = DictCredentials({"login": "first"})

    async def exercise():
        await first.bind(phone)
        wrapped = StealthDevice(RecordingDevice(phone))
        await provider.bind(wrapped)
        await provider.bind(phone)
        assert (
            dict(phone.extensions)
            == dict(wrapped.extensions)
            == {"vault": first, "second": provider}
        )
        await first.unbind()
        assert dict(phone.extensions) == {"second": provider}
        await provider.unbind()
        await provider.unbind()
        assert not phone.extensions
        if isinstance(provider, CloudCredentials):
            await provider.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AndroidDevice("local", credentials=[]),
        lambda: IOSDevice("http://local", credentials=[]),
        lambda: CloudDevice(api_key="fake", credentials=[]),
    ],
)
def test_device_credentials_constructor_is_removed(factory):
    with pytest.raises(
        TypeError, match="provider.bind|unexpected keyword argument 'credentials'"
    ):
        factory()


def test_mailbox_binding_is_local_and_supports_local_typing():
    phone = AndroidDevice("local")
    mailbox = CloudMailbox("mailbox", api_key="fake")
    mailbox.wait_for_code = AsyncMock(return_value="123456")

    async def exercise():
        await mailbox.bind(RecordingDevice(phone))
        assert phone.mailbox is mailbox
        assert await mailbox.wait_for_code() == "123456"
        await mailbox.unbind()
        with pytest.raises(DeviceOperationNotSupported, match="No email extension"):
            _ = phone.mailbox

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "connected,restarted", [(True, False), (False, False), (True, True)]
)
def test_phone_proxy_switch_checks_status_and_waits_after_restart(
    monkeypatch, connected, restarted
):
    from mobilerun.clients.cloud import proxies

    phone = CloudDevice("phone", api_key="fake", billing="paid")
    connect = AsyncMock()
    monkeypatch.setattr(proxies, "connect_device", connect)
    phone.info = AsyncMock(
        return_value=NS(state="restarting" if restarted else "ready")
    )
    phone.wait_ready = AsyncMock()
    from mobilerun.core.proxy import ProxyStatus

    phone._proxy_status = AsyncMock(return_value=ProxyStatus(connected=connected))
    proxy = CloudRotatingProxy(country="DE")

    async def exercise():
        if connected:
            await proxy.bind(phone, force=True)
            assert (await proxy.status()).connected
            assert phone.extensions["proxy"] is proxy
        else:
            with pytest.raises(ProxyNotActive):
                await proxy.bind(phone, force=True)
            assert not phone.extensions

    asyncio.run(exercise())
    assert phone.wait_ready.await_count == int(restarted)
    connect.assert_awaited_once()
    assert proxy.exclusive == "proxy"


def test_default_proxy_uses_smart_ip_setting(monkeypatch):
    monkeypatch.setenv("MOBILERUN_CLOUD_PROXY_SMART_IP", "false")
    assert CloudRotatingProxy._from_settings().smart_ip is False
