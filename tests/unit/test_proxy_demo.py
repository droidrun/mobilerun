import asyncio
from types import SimpleNamespace as Obj
from typing import cast
from unittest.mock import AsyncMock

import pytest

from tests.demos import proxy_demo as demo


@pytest.fixture
def setup_demo(monkeypatch):
    phone = Obj(
        proxy_status=AsyncMock(return_value=Obj(connected=True, name="old")),
        set_proxy=AsyncMock(return_value=Obj(connected=True, name="new")),
        close=AsyncMock(),
        stop=AsyncMock(),
        terminate=AsyncMock(),
    )
    phone.proxy = Obj(status=phone.proxy_status)
    monkeypatch.setattr(
        demo.proxies, "device_status", AsyncMock(side_effect=AssertionError("bypass"))
    )

    async def bind(proxy, device, **kwargs):
        assert kwargs == {"force": True}
        await device.set_proxy(proxy)
        device.proxy = proxy
        return proxy

    monkeypatch.setattr(demo.Proxy, "bind", bind)

    async def status(proxy):
        return Obj(connected=True, name="new")

    monkeypatch.setattr(demo.Proxy, "status", status)
    device = AsyncMock(return_value=phone)
    monkeypatch.setattr(demo, "CloudDevice", device)
    monkeypatch.setattr(
        demo.devices,
        "list_devices",
        AsyncMock(return_value=[Obj(name="tester1", id="phone")]),
    )
    monkeypatch.setattr(
        demo.proxies,
        "list_proxies",
        AsyncMock(
            return_value=Obj(
                items=[
                    Obj(id="old-id", name="old", country="US", status="active"),
                    Obj(id="new-id", name="new", country="DE", status="active"),
                ],
                pagination=Obj(has_next=False),
            )
        ),
    )
    return phone, device


def run(*args):
    return asyncio.run(demo.main(list(args)))


def test_selects_different_existing_proxy_and_closes_without_stopping(
    setup_demo, capsys
):
    phone, device = setup_demo
    assert run() == 0
    device.assert_awaited_once_with(device_id="phone", billing=demo.Billing.PAID)
    assert phone.set_proxy.await_args.args[0].id == "new-id"
    phone.close.assert_awaited_once()
    phone.stop.assert_not_awaited()
    phone.terminate.assert_not_awaited()
    assert "old" in capsys.readouterr().out


def test_missing_phone_lists_names(setup_demo, capsys):
    assert run("--name", "missing") == 1
    assert "tester1" in capsys.readouterr().err
    setup_demo[1].assert_not_awaited()


def test_explicit_proxy_id(setup_demo):
    assert run("--proxy-id", "new-id") == 0
    assert setup_demo[0].set_proxy.await_args.args[0].id == "new-id"


def test_no_alternative(setup_demo, capsys):
    cast(AsyncMock, demo.proxies.list_proxies).return_value.items.pop()
    assert run() == 1
    assert "alternative" in capsys.readouterr().err.lower()
    setup_demo[0].set_proxy.assert_not_awaited()
    setup_demo[0].close.assert_awaited_once()


def test_country_decline_never_switches(setup_demo, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert run("--country", "DE") == 1
    setup_demo[1].assert_not_awaited()


def test_country_yes_uses_paid(setup_demo):
    assert run("--country", "DE", "--yes") == 0
    setup_demo[1].assert_awaited_once_with(device_id="phone", billing=demo.Billing.PAID)
    assert setup_demo[0].set_proxy.await_args.args[0].country == "DE"


def test_unknown_current_proxy_fails_safely(setup_demo, capsys):
    setup_demo[0].proxy_status.return_value.name = None
    assert run() == 1
    assert "identify" in capsys.readouterr().err.lower()
    setup_demo[0].set_proxy.assert_not_awaited()


def test_proxy_not_active_reports_error_and_closes(setup_demo, capsys):
    setup_demo[0].set_proxy.side_effect = demo.ProxyNotActive("disconnected")
    assert run() == 1
    assert "not active" in capsys.readouterr().err
    setup_demo[0].close.assert_awaited_once()
