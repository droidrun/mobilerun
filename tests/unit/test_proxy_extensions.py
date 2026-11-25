"""Proxy extensions bind exclusively and switch explicitly."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.proxy import (
    CloudFixedProxy,
    CloudRotatingProxy,
    CloudSocks5Proxy,
    LocalProxy,
)
from mobilerun.core.proxy.target import ProxyTarget
from mobilerun.errors import ExtensionConflict
from mobilerun.models.proxy import ProxyStatus
from tests.support.devices import MemoryDevice
from tests.support.extensions import check_extension_contract


class FakeTarget(ProxyTarget):
    def __init__(self):
        self.current = ProxyStatus(False)
        self.connect = AsyncMock(side_effect=self._connect)
        self.disconnect_calls = 0

    async def _connect(self, name):
        self.current = ProxyStatus(True, name, "socks5")
        return self.current

    async def connect_fixed(self, *, id):
        return await self.connect(id)

    async def connect_rotating(self, *, country, smart_ip):
        return await self.connect(country)

    async def connect_socks5(self, host, port, user, password):
        return await self.connect(host)

    async def set_http_proxy(self, host, port):
        return await self.connect(f"{host}:{port}")

    async def status(self):
        return self.current

    async def disconnect(self):
        self.disconnect_calls += 1
        self.current = ProxyStatus(False)


class ProxyMemory(MemoryDevice):
    def __init__(self, cloud=False):
        super().__init__(
            cloud_id="cloud" if cloud else None,
            supported_operations=frozenset(
                {"proxy_switch"} if cloud else {"http_proxy"}
            ),
        )
        self.target = FakeTarget()

    @property
    def _proxy_target(self):
        return self.target


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
