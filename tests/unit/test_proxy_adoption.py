"""Read-only cloud proxy adoption and reconciliation of partial switches."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from test_proxy_extensions import ProxyMemory

from mobilerun.core.devices import CloudDevice
from mobilerun.core.proxy import CloudFixedProxy, CloudProxy
from mobilerun.errors import ExtensionConflict, ExtensionNotBound
from mobilerun.models.proxy import ProxyStatus


@pytest.fixture(autouse=True)
def isolated_number_loading(monkeypatch):
    from mobilerun.clients.cloud import devices

    monkeypatch.setattr(
        devices,
        "capabilities",
        AsyncMock(
            return_value=SimpleNamespace(capabilities=SimpleNamespace(telephony=False))
        ),
    )
    monkeypatch.setattr(CloudDevice, "_load_number", AsyncMock())


@pytest.mark.parametrize("name", ["mrc:known-id", "resolvable-reference"])
def test_existing_cloud_connect_adopts_known_proxy_without_mutation(monkeypatch, name):
    from mobilerun.clients.cloud import proxies

    status = AsyncMock(return_value=ProxyStatus(True, name, "socks5"))
    resolve = AsyncMock(return_value=SimpleNamespace(id="known-id"))
    switch, disconnect = AsyncMock(), AsyncMock()
    monkeypatch.setattr(proxies, "device_status", status)
    monkeypatch.setattr(proxies, "get_proxy", resolve)
    monkeypatch.setattr(proxies, "connect_device", switch)
    monkeypatch.setattr(proxies, "disconnect_device", disconnect)

    async def scenario():
        phone = CloudDevice("phone", api_key="test")
        phone._ensure_connected = AsyncMock()
        await asyncio.gather(phone.connect(), phone.connect())
        proxy = cast(CloudFixedProxy, phone.proxy)
        assert proxy.id == "known-id"
        assert proxy.device is phone
        assert phone.proxy is proxy
        await phone.connect()
        status.assert_awaited_once()
        assert (await proxy.status()).connected
        with pytest.raises(ExtensionConflict, match="cloud phones need a proxy"):
            await proxy.unbind()
        await phone.close()
        switch.assert_not_awaited()
        disconnect.assert_not_awaited()
        if name.startswith("mrc:"):
            resolve.assert_not_awaited()
        else:
            assert resolve.await_args.args == (name,)

    asyncio.run(scenario())


@pytest.mark.parametrize("name", ["", None, "unresolvable-reference"])
def test_existing_cloud_connect_adopts_unknown_proxy(monkeypatch, name):
    from mobilerun.clients.cloud import proxies
    from mobilerun.errors.cloud import CloudResourceNotFound

    status = AsyncMock(return_value=ProxyStatus(True, name, "socks5"))
    monkeypatch.setattr(proxies, "device_status", status)
    monkeypatch.setattr(
        proxies, "get_proxy", AsyncMock(side_effect=CloudResourceNotFound())
    )
    switch = AsyncMock()
    monkeypatch.setattr(proxies, "connect_device", switch)

    async def scenario():
        from mobilerun.core.proxy import ExistingCloudProxy

        phone = CloudDevice("phone", api_key="test")
        phone._ensure_connected = AsyncMock()
        await phone.connect()
        observed = cast(ExistingCloudProxy, phone.proxy)
        assert observed.exclusive == "proxy"
        assert observed.device is phone
        assert (await observed.status()).connected
        assert "unknown" in str((await observed.context())[0].data)
        with pytest.raises(ExtensionConflict, match="cloud phones need a proxy"):
            await observed.unbind()
        with pytest.raises(ExtensionConflict, match="unbind it first"):
            await CloudFixedProxy("new").bind(phone)
        switch.assert_not_awaited()
        await phone.close()

    asyncio.run(scenario())


def test_disconnected_cloud_connect_has_clear_missing_proxy_error(monkeypatch):
    from mobilerun.clients.cloud import proxies

    monkeypatch.setattr(
        proxies, "device_status", AsyncMock(return_value=ProxyStatus(False))
    )
    switch = AsyncMock()
    monkeypatch.setattr(proxies, "connect_device", switch)

    async def scenario():
        phone = CloudDevice("phone", api_key="test")
        phone._ensure_connected = AsyncMock()
        await phone.connect()
        assert not phone.extensions
        with pytest.raises(AttributeError, match=r"no active proxy.*bind"):
            _ = phone.proxy
        await phone.close()
        switch.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_kind", ["error", "timeout", "cancel"])
def test_partial_initial_switch_reconciles_before_original_error(failure_kind):
    async def scenario():
        phone = ProxyMemory(True)
        started = asyncio.Event()
        original = RuntimeError("backend changed state then failed")

        async def partial_switch(name):
            phone.target.current = ProxyStatus(True, "mrc:actually-active", "socks5")
            started.set()
            if failure_kind in ("cancel", "timeout"):
                await asyncio.Event().wait()
            raise original

        phone.target.connect.side_effect = partial_switch
        incoming = CloudFixedProxy("requested")

        async def bind():
            async with asyncio.timeout(0.05 if failure_kind == "timeout" else None):
                await incoming.bind(phone)

        task = asyncio.create_task(bind())
        await started.wait()
        if failure_kind == "cancel":
            task.cancel()
        error_type = {
            "error": RuntimeError,
            "timeout": TimeoutError,
            "cancel": asyncio.CancelledError,
        }[failure_kind]
        with pytest.raises(error_type) as raised:
            await task
        if failure_kind == "error":
            assert raised.value is original
        actual = cast(CloudFixedProxy, phone.proxy)
        assert actual.id == "actually-active"
        assert actual.device is phone
        assert incoming.device is None
        assert (await actual.status()).connected
        assert phone.target.connect.await_count == 1
        await phone.close()

    asyncio.run(scenario())


def test_partial_bind_failed_status_read_marks_uncertain_and_preserves_error():
    async def scenario():
        phone = ProxyMemory(True)
        original = RuntimeError("switch failed")
        phone.target.connect.side_effect = original
        phone.target.status = AsyncMock(
            side_effect=[ProxyStatus(False), RuntimeError("status failed")]
        )
        incoming = CloudFixedProxy("requested")
        with pytest.raises(RuntimeError) as raised:
            await incoming.bind(phone)
        assert raised.value is original
        assert any("uncertain" in note for note in original.__notes__)
        assert phone.extensions["proxy"] is incoming
        with pytest.raises(ExtensionNotBound, match="uncertain"):
            await incoming.status()
        await phone.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "remote",
    [ProxyStatus(True, "mrc:actual"), ProxyStatus(True, ""), ProxyStatus(False)],
)
def test_failed_force_rollback_adopts_actual_remote_state(remote):
    async def scenario():
        phone = ProxyMemory(True)
        old = CloudFixedProxy("old")
        await old.bind(phone)
        original = RuntimeError("switch failed")

        async def fail(name):
            phone.target.current = remote
            raise original

        phone.target.connect.side_effect = fail
        incoming = CloudFixedProxy("new")
        with pytest.raises(RuntimeError) as raised:
            await incoming.bind(phone, force=True)
        assert raised.value is original
        assert old.device is None
        assert incoming.device is None
        if remote.connected:
            adopted = cast(CloudProxy, phone.proxy)
            assert adopted.device is phone
            if remote.name:
                assert cast(CloudFixedProxy, phone.proxy).id == "actual"
            else:
                from mobilerun.core.proxy import ExistingCloudProxy

                assert cast(ExistingCloudProxy, phone.proxy) is adopted
            assert not adopted._binding.uncertain
        else:
            assert not phone.extensions
        await phone.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("remote_name", ["mrc:old", "old", ""])
def test_successful_force_rollback_preserves_original_matching_or_unknown_proxy(
    remote_name,
):
    async def scenario():
        phone = ProxyMemory(True)
        old = CloudFixedProxy("old")
        await old.bind(phone, close_on_device_close=True)
        original_binding = old._binding
        original = RuntimeError("switch response lost")

        async def connect(name):
            phone.target.current = ProxyStatus(True, remote_name)
            if name == "new":
                raise original
            return ProxyStatus(True, "old")

        phone.target.connect.side_effect = connect
        incoming = CloudFixedProxy("new")
        with pytest.raises(RuntimeError) as raised:
            await incoming.bind(phone, force=True)
        assert raised.value is original
        assert phone.proxy is old
        assert old._binding is original_binding
        assert old._binding.owned
        assert not old._binding.uncertain
        assert old.device is phone
        assert incoming.device is None
        await phone.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_readback", [False, True])
def test_successful_force_rollback_still_adopts_remote_truth(cancel_readback):
    async def scenario():
        phone = ProxyMemory(True)
        old = CloudFixedProxy("old")
        await old.bind(phone)
        original = RuntimeError("switch response lost")

        async def connect(name):
            phone.target.current = ProxyStatus(True, "mrc:actual")
            if name == "new":
                raise original
            # Rollback returns success, but remote truth still differs.
            return ProxyStatus(True, "mrc:old")

        phone.target.connect.side_effect = connect
        readback_started, finish_readback = asyncio.Event(), asyncio.Event()
        original_reconcile = old._reconcile_bind_failure

        async def readback(device, failure):
            readback_started.set()
            await finish_readback.wait()
            await original_reconcile(device, failure)

        reconcile = AsyncMock(side_effect=readback)
        old._reconcile_bind_failure = reconcile
        incoming = CloudFixedProxy("new")
        switching = asyncio.create_task(incoming.bind(phone, force=True))
        await readback_started.wait()
        if cancel_readback:
            for _ in range(2):
                switching.cancel()
                await asyncio.sleep(0)
            assert not switching.done()
        finish_readback.set()
        with pytest.raises(RuntimeError) as raised:
            await switching
        assert raised.value is original
        reconcile.assert_awaited_once_with(phone, original)
        assert cast(CloudFixedProxy, phone.proxy).id == "actual"
        assert not phone.proxy._binding.uncertain
        assert old.device is None
        assert incoming.device is None
        await phone.close()

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_unknown_proxy_satisfies_public_contract():
    from mobilerun.core.proxy import ExistingCloudProxy
    from tests.support.extensions import check_extension_contract

    async def device():
        phone = ProxyMemory(True)
        phone.target.current = ProxyStatus(True, "")
        return phone

    async def incompatible():
        return ProxyMemory(False)

    asyncio.run(
        check_extension_contract(
            lambda name: ExistingCloudProxy(name=name),
            device,
            incompatible_device=incompatible,
            unbind_requires_replacement=True,
        )
    )


def test_repeated_cancel_waits_for_reconciliation_and_preserves_first_cancel():
    async def scenario():
        phone = ProxyMemory(True)
        switched, reading, release_read = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )
        status_calls = 0

        async def switch(name):
            switched.set()
            await asyncio.Event().wait()

        async def status():
            nonlocal status_calls
            status_calls += 1
            if status_calls == 1:
                return ProxyStatus(False)
            reading.set()
            await release_read.wait()
            return ProxyStatus(True, "mrc:actual")

        phone.target.connect.side_effect = switch
        phone.target.status = status
        task = asyncio.create_task(CloudFixedProxy("requested").bind(phone))
        await switched.wait()
        task.cancel("first cancellation")
        await reading.wait()
        task.cancel("second cancellation")
        release_read.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert raised.value.args == ("first cancellation",)
        assert cast(CloudFixedProxy, phone.proxy).id == "actual"
        await phone.close()

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_connect_recovers_uncertain_proxy_without_switching(monkeypatch):
    from mobilerun.clients.cloud import proxies

    original = RuntimeError("switch failed")
    status = AsyncMock(
        side_effect=[
            ProxyStatus(False),
            RuntimeError("status unavailable"),
            ProxyStatus(True, "mrc:actual"),
        ]
    )
    switch = AsyncMock(side_effect=original)
    monkeypatch.setattr(proxies, "device_status", status)
    monkeypatch.setattr(proxies, "connect_device", switch)

    async def scenario():
        phone = CloudDevice("phone", api_key="test")
        phone._ensure_connected = AsyncMock()
        with pytest.raises(RuntimeError) as raised:
            await CloudFixedProxy("requested").bind(phone)
        assert raised.value is original
        with pytest.raises(ExtensionNotBound, match="uncertain"):
            await phone.proxy.status()
        await phone.connect()
        assert cast(CloudFixedProxy, phone.proxy).id == "actual"
        assert not phone.extensions["proxy"]._binding.uncertain
        switch.assert_awaited_once()
        await phone.close()

    asyncio.run(scenario())


def test_unknown_proxy_is_replaced_only_with_force_without_disconnect():
    from mobilerun.core.proxy import ExistingCloudProxy

    async def scenario():
        phone = ProxyMemory(True)
        phone.target.current = ProxyStatus(True, "")
        observed = ExistingCloudProxy()
        await observed.bind(phone)
        incoming = CloudFixedProxy("new")
        with pytest.raises(ExtensionConflict, match="unbind it first"):
            await incoming.bind(phone)
        phone.target.connect.assert_not_awaited()
        await incoming.bind(phone, force=True)
        assert phone.proxy is incoming
        assert observed.device is None
        assert incoming.device is phone
        assert phone.target.connect.await_count == 1
        assert phone.target.disconnect_calls == 0
        await phone.close()

    asyncio.run(scenario())


def test_existing_phone_failed_status_read_can_be_retried(monkeypatch):
    from mobilerun.clients.cloud import proxies

    unavailable = RuntimeError("status unavailable")
    status = AsyncMock(side_effect=[unavailable, ProxyStatus(True, "mrc:actual")])
    monkeypatch.setattr(proxies, "device_status", status)

    async def scenario():
        phone = CloudDevice("phone", api_key="test")
        phone._ensure_connected = AsyncMock()
        with pytest.raises(RuntimeError) as raised:
            await phone.connect()
        assert raised.value is unavailable
        assert not phone.extensions
        await phone.connect()
        assert cast(CloudFixedProxy, phone.proxy).id == "actual"
        await phone.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("lookup_timeout", [False, True])
def test_failed_identity_lookup_adopts_known_active_unknown_instead_of_uncertain(
    monkeypatch, lookup_timeout
):
    from mobilerun.core.proxy import ExistingCloudProxy, adoption

    monkeypatch.setattr(adoption, "_READBACK_TIMEOUT", 0.02)

    async def scenario():
        phone = ProxyMemory(True)
        original = RuntimeError("partial switch failed")

        async def switch(name):
            phone.target.current = ProxyStatus(True, "unresolved-alias")
            raise original

        phone.target.connect.side_effect = switch

        async def resolve(reference):
            if lookup_timeout:
                await asyncio.Event().wait()
            raise RuntimeError("identity lookup failed")

        phone.target._resolve_reference = resolve
        requested = CloudFixedProxy("requested")
        with pytest.raises(RuntimeError) as raised:
            await requested.bind(phone)
        assert raised.value is original
        actual = cast(ExistingCloudProxy, phone.proxy)
        assert actual.device is phone
        assert not actual._binding.uncertain
        assert requested.device is None
        assert (await actual.status()).connected
        await phone.close()

    asyncio.run(scenario())
