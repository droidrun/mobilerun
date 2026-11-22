import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.devices.android import AndroidDevice
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.proxy import (
    CloudFixedProxy,
    CloudRotatingProxy,
    CloudSocks5Proxy,
    Proxy,
)
from mobilerun.models.billing import Billing


@pytest.mark.parametrize("host", ["a>/x", "$(id)", "a b", "a<b", "a*b", "a#b"])
def test_proxy_address_rejects_shell_metacharacters(host):
    from mobilerun.core.proxy import LocalProxy

    with pytest.raises(ValueError):
        LocalProxy(host, 8080)


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_proxy_address_rejects_invalid_ports(port):
    from mobilerun.core.proxy import LocalProxy

    with pytest.raises(ValueError):
        LocalProxy("example.org", port)


def test_proxy_values_and_redaction():
    assert CloudRotatingProxy(country="DE")._parameters() == {
        "connect": {"country": "DE"},
        "smart_ip": True,
    }
    assert CloudFixedProxy(id="id")._parameters() == {"connect": {"id": "id"}}
    proxy = CloudSocks5Proxy("host", 1080, "user", "secret")
    assert proxy._parameters() == {
        "socks5": {"host": "host", "port": 1080, "user": "user", "password": "secret"}
    }
    assert "secret" not in repr(proxy)
    with pytest.raises(ValueError):
        CloudRotatingProxy(country="")
    assert not hasattr(Proxy, "connect") and not hasattr(Proxy, "socks5")


def test_auto_create_with_proxy_and_default(monkeypatch):
    monkeypatch.setenv("MOBILERUN_BILLING", "paid")
    from mobilerun_core_local.driver import cloud as driver

    devices = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(id="one", name="one")),
        wait_ready=AsyncMock(),
        retrieve_capabilities=AsyncMock(
            return_value=SimpleNamespace(capabilities=SimpleNamespace(telephony=False))
        ),
    )
    client = SimpleNamespace(devices=devices, with_options=lambda **kw: client)
    monkeypatch.setattr(driver, "AsyncMobilerun", lambda **kw: client)
    monkeypatch.setenv("MOBILERUN_CLOUD_PROXY_COUNTRY", "DE")
    device = CloudDevice(
        api_key="test",
        proxy=CloudRotatingProxy(country="FR"),
        country="US",
        locale="en-US",
        timezone="UTC",
    )
    asyncio.run(device.connect())
    assert device.extensions["proxy"] is device._creation_proxy
    assert device.__getattr__("proxy").device is device
    devices.create.assert_awaited_once_with(
        device_type="android_cloud_phone",
        proxy={"connect": {"country": "FR"}, "smart_ip": True},
        body_country="US",
        locale="en-US",
        timezone="UTC",
        billing="auto",
    )
    devices.create.reset_mock()
    asyncio.run(CloudDevice(api_key="test").connect())
    assert devices.create.await_args is not None
    assert devices.create.await_args.kwargs["proxy"] == {
        "connect": {"country": "DE"},
        "smart_ip": True,
    }


def test_concurrent_cloud_creation_publishes_once(monkeypatch):
    from mobilerun_core_local.driver import cloud as driver

    async def create(**kwargs):
        await asyncio.sleep(0)
        return SimpleNamespace(id="one", name="one")

    devices = SimpleNamespace(
        create=AsyncMock(side_effect=create),
        wait_ready=AsyncMock(),
        retrieve_capabilities=AsyncMock(
            return_value=SimpleNamespace(capabilities=SimpleNamespace(telephony=False))
        ),
    )
    client = SimpleNamespace(devices=devices, with_options=lambda **kw: client)
    monkeypatch.setattr(driver, "AsyncMobilerun", lambda **kw: client)
    proxy = CloudRotatingProxy("FR")
    phone = CloudDevice(api_key="test", proxy=proxy)

    async def scenario():
        await asyncio.gather(phone.connect(), phone.connect())
        assert phone.extensions["proxy"] is proxy
        assert proxy.device is phone
        devices.create.assert_awaited_once()

    asyncio.run(scenario())


def test_running_proxy_and_local_commands(monkeypatch):
    from mobilerun_core_local.driver import cloud as driver

    from mobilerun.clients.cloud import proxies

    endpoint = SimpleNamespace(
        connect=AsyncMock(),
        disconnect=AsyncMock(),
        status=AsyncMock(
            return_value=SimpleNamespace(
                connected=True, name="proxy", protocol="socks5"
            )
        ),
    )
    client = SimpleNamespace(
        devices=SimpleNamespace(
            proxy=endpoint,
            retrieve=AsyncMock(return_value=SimpleNamespace(state="ready", name="one")),
        ),
        with_options=lambda **kw: client,
        api_key="test",
        base_url="https://example.test",
    )
    monkeypatch.setattr(driver, "AsyncMobilerun", lambda **kw: client)
    context = AsyncMock()
    context.__aenter__.return_value = client
    monkeypatch.setattr(proxies, "cloud_client", lambda *args, **kwargs: context)

    async def run():
        device = CloudDevice("one", api_key="test")
        proxy = CloudFixedProxy(id="p")
        await proxy.bind(device, force=True)
        assert device.extensions["proxy"] is proxy
        assert (await proxy.status()).connected
        endpoint.connect.assert_awaited_once_with("one", connect={"id": "p"})
        local = AndroidDevice(serial="emulator-5554")
        adb = SimpleNamespace(shell=AsyncMock(return_value="example.org:8080"))
        monkeypatch.setattr(local, "_adb_device", adb)
        local._driver._connected = True
        from mobilerun.core.proxy import LocalProxy

        await LocalProxy("example.org", 8080).bind(local)
        replacement = LocalProxy("::1", 8888)
        await replacement.bind(local, force=True)
        await replacement.unbind()
        assert [call.args[0] for call in adb.shell.await_args_list] == [
            "settings put global http_proxy example.org:8080",
            "settings get global http_proxy",
            "settings put global http_proxy :0",
            "settings put global http_proxy '[::1]:8888'",
            "settings get global http_proxy",
            "settings put global http_proxy :0",
        ]

    asyncio.run(run())


def test_paid_setting(monkeypatch):
    from mobilerun.config.settings import get_settings

    monkeypatch.setenv("MOBILERUN_BILLING", "paid")
    assert get_settings().MOBILERUN_BILLING == "paid"


def test_connect_proxy_helpers_and_paid_guard(monkeypatch):
    from mobilerun.clients.cloud import proxies

    endpoint = SimpleNamespace(
        list=AsyncMock(return_value="list"),
        retrieve=AsyncMock(return_value="get"),
        ensure_healthy=AsyncMock(),
        ping=AsyncMock(),
        buy=AsyncMock(),
    )
    client = SimpleNamespace(connect=SimpleNamespace(proxies=endpoint))
    client.__aenter__ = AsyncMock(return_value=client)

    class Context:
        async def __aenter__(self):
            return client

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(proxies, "cloud_client", lambda *args, **kwargs: Context())

    async def run():
        assert (
            await proxies.list_proxies(billing=Billing.INCLUDED, country="DE") == "list"
        )
        assert await proxies.get_proxy("one", billing=Billing.INCLUDED) == "get"
        with pytest.raises(ValueError):
            await proxies.buy_proxy("DE", billing=cast(Billing, None))
        endpoint.buy.assert_not_awaited()
        await proxies.buy_proxy("DE", billing=Billing.PAID)
        endpoint.buy.assert_awaited_once_with(country="DE", type="residential")

    asyncio.run(run())


@pytest.mark.parametrize("billing", ["no_commerce", "included", "paid"])
def test_connect_country_guard_and_existing_id(monkeypatch, billing):
    from mobilerun.clients.cloud import proxies

    endpoint = SimpleNamespace(connect=AsyncMock())
    sdk = SimpleNamespace(devices=SimpleNamespace(proxy=endpoint))
    context = AsyncMock()
    context.__aenter__.return_value = sdk
    monkeypatch.setattr(proxies, "cloud_client", lambda *args, **kwargs: context)

    async def exercise():
        await proxies.connect_device(
            "phone", {"connect": {"id": "existing"}}, billing=billing
        )
        endpoint.connect.reset_mock()
        if billing == "paid":
            await proxies.connect_device(
                "phone", {"connect": {"country": "DE"}}, billing=Billing(billing)
            )
            endpoint.connect.assert_awaited_once_with(
                "phone", connect={"country": "DE"}
            )
        else:
            with pytest.raises(
                ValueError, match="billing='paid'.*MOBILERUN_BILLING=paid"
            ):
                await proxies.connect_device(
                    "phone", {"connect": {"country": "DE"}}, billing=billing
                )
            endpoint.connect.assert_not_awaited()

    asyncio.run(exercise())


def test_cloud_phone_defaults_to_rotating_proxy_and_rejects_none(monkeypatch):
    from mobilerun.core.proxy import CloudRotatingProxy

    monkeypatch.delenv("MOBILERUN_CLOUD_PROXY_COUNTRY", raising=False)
    proxy = CloudRotatingProxy._from_settings()
    assert proxy.country == "us" and proxy.smart_ip is True
    device = CloudDevice(api_key="test")
    assert device._create_options["proxy"] == proxy._parameters()
    assert device._creation_proxy._parameters() == proxy._parameters()
    with pytest.raises(ValueError, match="proxy"):
        CloudDevice(api_key="test", proxy=None)
    device = CloudDevice(api_key="test", proxy=proxy)
    assert device._create_options["proxy"] == {
        "connect": {"country": "us"},
        "smart_ip": True,
    }
