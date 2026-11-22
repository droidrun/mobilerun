import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.devices import AndroidDevice, CloudDevice, Device, DevicePool
from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.proxy import (
    CloudFixedProxy,
    CloudRotatingProxy,
    CloudSocks5Proxy,
    LocalProxy,
    ProxyStatus,
)
from mobilerun.errors import ExtensionConflict, ExtensionNotSupported


@pytest.mark.parametrize(
    "proxy",
    [
        CloudFixedProxy(id="p"),
        CloudRotatingProxy(country="US"),
        CloudSocks5Proxy("host", 1080),
    ],
)
def test_local_rejects_non_http_without_connecting(proxy):
    device = AndroidDevice(serial="local")
    with pytest.raises(ExtensionNotSupported, match="cloud device"):
        asyncio.run(proxy.bind(device))


def test_cloud_rejects_http_without_sdk_calls():
    device = CloudDevice("cloud", api_key="test")
    with pytest.raises(
        ExtensionNotSupported,
        match="local Android",
    ):
        asyncio.run(LocalProxy("host", 8080).bind(device))


def test_http_application_status_and_clear(monkeypatch):
    async def run():
        phone = AndroidDevice(serial="local")
        phone._ensure_connected = AsyncMock()
        commands: list[str] = []
        value = "null"

        async def shell(command, **kwargs):
            nonlocal value
            commands.append(command)
            if command.startswith("settings put global http_proxy "):
                value = command.removeprefix("settings put global http_proxy ").strip(
                    "'"
                )
            return value

        monkeypatch.setattr(phone, "_adb_device", SimpleNamespace(shell=shell))
        proxy = LocalProxy("::1", 8888)
        await proxy.bind(phone)
        assert await proxy.status() == ProxyStatus(True, "[::1]:8888")
        assert phone.extensions["proxy"] is proxy
        await proxy.unbind()
        assert not phone.extensions
        assert await phone._proxy_target.status() == ProxyStatus(False)
        assert "settings put global http_proxy '[::1]:8888'" in commands

    asyncio.run(run())


@pytest.mark.parametrize(
    ("proxy", "params"),
    [
        (CloudFixedProxy(id="managed"), {"connect": {"id": "managed"}}),
        (
            CloudRotatingProxy(country="US"),
            {"connect": {"country": "US"}, "smart_ip": True},
        ),
        (
            CloudSocks5Proxy("host", 1080, "user", "password"),
            {
                "socks5": {
                    "host": "host",
                    "port": 1080,
                    "user": "user",
                    "password": "password",
                }
            },
        ),
    ],
)
def test_cloud_proxy_types_status_and_clear(monkeypatch, proxy, params):
    from mobilerun.clients.cloud import proxies

    async def run():
        phone = CloudDevice("cloud", api_key="test")
        connect = AsyncMock()
        disconnect = AsyncMock()
        monkeypatch.setattr(proxies, "connect_device", connect)
        monkeypatch.setattr(proxies, "disconnect_device", disconnect)
        monkeypatch.setattr(
            proxies,
            "device_status",
            AsyncMock(
                return_value=SimpleNamespace(
                    connected=True, name="active", protocol="socks5"
                )
            ),
        )
        monkeypatch.setattr(
            phone, "info", AsyncMock(return_value=SimpleNamespace(state="ready"))
        )
        await proxy.bind(phone, force=True)
        assert await proxy.status() == ProxyStatus(True, "active", "socks5")
        assert connect.await_args is not None
        assert connect.await_args.args == ("cloud", params)
        assert phone.extensions["proxy"] is proxy
        with pytest.raises(ExtensionConflict, match="cloud phones need a proxy"):
            await proxy.unbind()
        await phone.close()
        assert not phone.extensions
        disconnect.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize(
    "method,args",
    [
        ("set_proxy", (LocalProxy("host", 80),)),
        ("clear_proxy", ()),
        ("proxy_status", ()),
    ],
)
def test_removed_device_proxy_operations(method, args):
    with pytest.raises(AttributeError, match=method):
        getattr(Device(), method)


@pytest.mark.parametrize(
    ("proxy", "payload"),
    [
        (CloudRotatingProxy("US"), {"connect": {"country": "US"}, "smartIp": True}),
        (
            CloudRotatingProxy("DE", smart_ip=False),
            {"connect": {"country": "DE"}, "smartIp": False},
        ),
        (CloudFixedProxy("existing"), {"connect": {"id": "existing"}}),
    ],
)
def test_managed_proxy_sdk_wire_payloads(monkeypatch, proxy, payload):
    import json
    from contextlib import nullcontext
    from typing import cast

    import httpx

    from mobilerun.clients.cloud import devices, proxies
    from mobilerun.clients.cloud.client import AsyncMobilerun
    from mobilerun.core.proxy.target import CloudProxyTarget
    from mobilerun.models import Billing

    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "phone", "name": None})

    async def run():
        async with AsyncMobilerun(
            api_key="test",
            base_url="https://sdk.test/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            monkeypatch.setattr(
                proxies, "cloud_client", lambda *args, **kwargs: nullcontext(client)
            )
            await devices.create(
                client=client,
                billing=Billing.PAID,
                proxy=cast(devices.Proxy, proxy._parameters()),
            )

            async def connect(params):
                await proxies.connect_device("phone", params, billing=Billing.PAID)
                return ProxyStatus(True)

            await proxy._apply(CloudProxyTarget(connect))

    asyncio.run(run())
    assert seen == [{"proxy": payload}, payload]


def test_disconnect_conflict_is_clear_not_retried_and_preserves_selection(monkeypatch):
    from contextlib import nullcontext

    import httpx

    from mobilerun.clients.cloud import proxies
    from mobilerun.clients.cloud.client import AsyncMobilerun
    from mobilerun.errors import CloudAPIError

    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(409, json={"detail": "No devices available"})

    async def run():
        async with AsyncMobilerun(
            api_key="test",
            base_url="https://sdk.test/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            monkeypatch.setattr(
                proxies, "cloud_client", lambda *args, **kwargs: nullcontext(client)
            )
            phone = CloudDevice("phone", api_key="test")
            with pytest.raises(
                CloudAPIError, match="HTTP 409.*backend could not disconnect.*no retry"
            ):
                await phone._proxy_target.disconnect()
            assert not phone.extensions

    asyncio.run(run())
    assert len(seen) == 1


def test_old_managed_proxy_class_is_not_exported():
    import mobilerun.core.proxy as proxies

    assert not hasattr(proxies, "ConnectProxy")
    assert not hasattr(CloudFixedProxy("existing"), "smart_ip")
    assert not hasattr(CloudSocks5Proxy("host", 1080), "smart_ip")


def test_wrapper_pool_proxy_operations():
    from mobilerun.core.proxy.target import AndroidProxyTarget
    from tests.support.devices import MemoryDevice

    class ProxyDevice(MemoryDevice):
        def __init__(self):
            super().__init__(supported_operations=frozenset({"http_proxy"}))
            self.current = ProxyStatus(False)

        @property
        def _proxy_target(self):
            return AndroidProxyTarget(self.apply, self.status, self.disconnect)

        async def apply(self, host, port):
            self.current = ProxyStatus(True, f"{host}:{port}")
            return self.current

        async def status(self):
            return self.current

        async def disconnect(self):
            self.current = ProxyStatus(False)

    async def run():
        pool = DevicePool([DeviceWrapper(ProxyDevice()), ProxyDevice()])
        await pool.bind_each(lambda _: LocalProxy("host", 80))
        assert (
            await pool.actions.call("proxy.status", {})
            == [ProxyStatus(True, "host:80")] * 2
        )
        await pool.unbind_extension("proxy")
        assert not any(pool.extensions)

    asyncio.run(run())
