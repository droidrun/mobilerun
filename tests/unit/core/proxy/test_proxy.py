"""Proxy binding, device operations, billing guards and SDK wire contracts."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.devices import AndroidDevice, CloudDevice, Device, DevicePool
from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.proxy import (
    CloudFixedProxy,
    CloudRotatingProxy,
    CloudSocks5Proxy,
    LocalProxy,
    Proxy,
    ProxyStatus,
)
from mobilerun.errors import ExtensionConflict, ExtensionNotSupported
from mobilerun.models.billing import Billing
from tests.support.devices import MemoryDevice
from tests.support.extensions import check_extension_contract
from tests.support.proxy import ProxyMemory


@pytest.mark.parametrize(
    "factory,cloud",
    [
        (lambda name: LocalProxy("localhost", 8080, name=name), False),
        (lambda name: CloudFixedProxy("proxy-id", name=name), True),
        (lambda name: CloudRotatingProxy("US", name=name), True),
        (lambda name: CloudSocks5Proxy("host", 1080, name=name), True),
    ],
)
def test_proxy_contract(factory, cloud):
    async def device():
        return ProxyMemory(cloud)

    async def incompatible():
        return MemoryDevice(cloud_id=None if cloud else "cloud")

    asyncio.run(
        check_extension_contract(
            factory,
            device,
            incompatible_device=incompatible,
            unbind_requires_replacement=cloud,
        )
    )


def test_cloud_proxy_force_switch_and_required_proxy():
    async def scenario():
        phone = ProxyMemory(True)
        first = CloudFixedProxy("first")
        second = CloudFixedProxy("second")
        await first.bind(phone)
        assert phone.__getattr__("proxy") is first
        with pytest.raises(ExtensionConflict, match="unbind it first"):
            await second.bind(phone)
        assert phone.target.connect.await_count == 1
        with pytest.raises(ExtensionConflict, match="cloud phones need a proxy"):
            await first.unbind()
        await second.bind(phone, force=True)
        assert first.device is None
        assert phone.extensions["proxy"] is second
        assert phone.target.connect.await_count == 2
        assert phone.target.disconnect_calls == 0
        await phone.close()

    asyncio.run(scenario())


def test_cloud_unmanaged_proxy_needs_force():
    async def scenario():
        phone = ProxyMemory(True)
        phone.target.current = ProxyStatus(True, "unmanaged")
        proxy = CloudFixedProxy("new")
        from mobilerun.core.proxy import UnmanagedProxyConflict

        with pytest.raises(UnmanagedProxyConflict, match="unmanaged"):
            await proxy.bind(phone)
        assert "proxy" not in phone.extensions
        assert proxy.device is None
        assert phone.target.connect.await_count == 0
        await proxy.bind(phone, force=True)
        assert phone.extensions["proxy"] is proxy
        await phone.close()

    asyncio.run(scenario())


def test_failed_force_restores_old_or_marks_it_uncertain():
    async def scenario(restore_fails):
        phone = ProxyMemory(True)
        old = CloudFixedProxy("old")
        new = CloudFixedProxy("new")
        await old.bind(phone)
        failure = RuntimeError("backend failed")
        phone.target.connect.side_effect = [
            failure,
            failure if restore_fails else ProxyStatus(True, "old"),
        ]
        if restore_fails:
            phone.target.status = AsyncMock(
                side_effect=[
                    phone.target.current,
                    phone.target.current,
                    RuntimeError("status read failed"),
                ]
            )
        with pytest.raises(RuntimeError) as raised:
            await new.bind(phone, force=True)
        assert raised.value is failure
        assert new.device is None
        if restore_fails:
            assert phone.extensions["proxy"] is old
            assert old._binding.uncertain
        else:
            assert phone.proxy is old
            assert not old._binding.uncertain
        phone.target.connect.side_effect = phone.target._connect
        await phone.close()

    asyncio.run(scenario(False))
    asyncio.run(scenario(True))


def test_force_does_not_release_owned_old_proxy():
    async def scenario():
        phone = ProxyMemory(True)
        old = CloudFixedProxy("old")
        old.on_close = AsyncMock()
        await old.bind(phone, close_on_device_close=True)
        new = CloudFixedProxy("new")
        await new.bind(phone, force=True)
        old.on_close.assert_not_awaited()
        assert old.device is None
        await phone.close()

    asyncio.run(scenario())


def test_unmanaged_refusal_does_not_reconcile_before_forced_retry():
    async def scenario():
        from mobilerun.core.proxy import UnmanagedProxyConflict

        phone = ProxyMemory(True)
        phone.target.current = ProxyStatus(True, "unmanaged")
        original = phone.target.status
        phone.target.status = AsyncMock(
            side_effect=[
                phone.target.current,
                RuntimeError("readback failed"),
                phone.target.current,
            ]
        )
        proxy = CloudFixedProxy("new")
        with pytest.raises(UnmanagedProxyConflict):
            await proxy.bind(phone)
        assert proxy._binding is None
        phone.target.status = AsyncMock(side_effect=original)
        await proxy.bind(phone, force=True)
        assert phone.extensions["proxy"] is proxy
        await phone.close()

    asyncio.run(scenario())


def test_cli_accepts_after_unmanaged_refusal_without_reconciliation(monkeypatch):
    from click.testing import CliRunner

    from mobilerun.cli.base import menu
    from mobilerun.cli.proxy import group

    phone = ProxyMemory(True)
    phone.target.current = ProxyStatus(True, "unmanaged")
    phone.target.status = AsyncMock(
        side_effect=[phone.target.current, RuntimeError("unexpected readback")]
    )
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(group, "CloudDevice", lambda **kwargs: phone)

    async def accept(message, *, default):
        assert default is False
        assert phone.target.status.await_count == 1
        phone.target.status = AsyncMock(return_value=phone.target.current)
        return True

    monkeypatch.setattr(group, "confirm_prompt_async", accept)
    result = CliRunner().invoke(group.proxy, ["bind", "new", "phone"])
    assert result.exit_code == 0, result.output
    phone.target.connect.assert_awaited_once_with("new")


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

    import httpx

    from mobilerun.clients.cloud import devices, proxies
    from mobilerun.clients.cloud.client import AsyncMobilerun
    from mobilerun.core.proxy.target import CloudProxyTarget

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


@pytest.mark.parametrize("host", ["a>/x", "$(id)", "a b", "a<b", "a*b", "a#b"])
def test_proxy_address_rejects_shell_metacharacters(host):

    with pytest.raises(ValueError):
        LocalProxy(host, 8080)


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_proxy_address_rejects_invalid_ports(port):

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
            return_value=SimpleNamespace(
                capabilities=SimpleNamespace(telephony=False, proxy=True)
            )
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
            return_value=SimpleNamespace(
                capabilities=SimpleNamespace(telephony=False, proxy=True)
            )
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
