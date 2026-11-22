"""Every public cloud boundary requires a commerce declaration before transport."""

import asyncio
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from mobilerun.clients.cloud import (
    apps,
    devices,
    files,
    mailboxes,
    numbers,
    proxies,
    secrets,
    tasks,
)
from mobilerun.config.settings import get_settings
from mobilerun.errors import BillingError
from mobilerun.models.billing import Billing


def _boundaries():
    root = Path(__file__).resolve().parents[2] / "mobilerun/clients/cloud"
    for path in sorted(root.glob("*.py")):
        if path.stem in {"checkout", "errors", "payments"}:
            continue  # Helpers are not transport or billing boundaries.
        module = importlib.import_module(f"mobilerun.clients.cloud.{path.stem}")
        for name, function in vars(module).items():
            if (
                not name.startswith("_")
                and name != "requires"
                and inspect.isfunction(function)
                and function.__module__ == module.__name__
            ):
                yield pytest.param(function, id=f"{path.stem}.{name}")


class _NoHTTP:
    def __getattr__(self, name):
        raise AssertionError(f"HTTP attempted: {name}")


def _arguments(function):
    args, kwargs = [], {}
    for param in inspect.signature(function).parameters.values():
        if param.name == "billing" or param.default is not inspect.Parameter.empty:
            continue
        if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            continue
        value = _NoHTTP() if param.name == "client" else "test"
        if param.kind is param.KEYWORD_ONLY:
            kwargs[param.name] = value
        else:
            args.append(value)
    return args, kwargs


@pytest.mark.parametrize("function", list(_boundaries()))
def test_missing_billing_fails_before_transport(function):
    args, kwargs = _arguments(function)
    with pytest.raises(TypeError, match="billing"):
        function(*args, **kwargs)


@pytest.mark.parametrize("function", list(_boundaries()))
def test_invalid_billing_fails_before_transport(function):
    args, kwargs = _arguments(function)
    with pytest.raises(
        BillingError, match=f"{function.__name__}.*given.*requires.*MOBILERUN_BILLING"
    ):
        function(*args, billing="invalid", **kwargs)


@pytest.mark.parametrize(
    "function",
    [devices.create, numbers.create, mailboxes.create, tasks.run, tasks.run_task],
)
def test_included_functions_refuse_no_commerce_before_transport(function):
    args, kwargs = _arguments(function)
    with pytest.raises(BillingError, match="requires billing='included'"):
        function(*args, billing=Billing.NO_COMMERCE, **kwargs)


@pytest.mark.parametrize("mode", [Billing.NO_COMMERCE, Billing.INCLUDED])
def test_paid_proxy_refuses_lower_modes_before_transport(mode):
    with pytest.raises(BillingError, match="buy_proxy.*requires billing='paid'"):
        proxies.buy_proxy("US", billing=mode)


class _TransportReached(BaseException):
    pass


class _FreeClient:
    def __getattr__(self, name):
        raise _TransportReached


@pytest.mark.parametrize("function", list(_boundaries()))
def test_every_free_boundary_accepts_no_commerce(function, monkeypatch, tmp_path):
    module = importlib.import_module(function.__module__)
    if function in (
        devices.create,
        numbers.create,
        mailboxes.create,
        tasks.run,
        tasks.run_task,
        proxies.buy_proxy,
    ):
        return

    def transport(*args, **kwargs):
        raise _TransportReached

    if hasattr(module, "cloud_client"):
        monkeypatch.setattr(module, "cloud_client", transport)
    if hasattr(module, "AsyncMobilerun"):
        monkeypatch.setattr(module, "AsyncMobilerun", transport)
    if hasattr(module, "_request"):
        monkeypatch.setattr(module, "_request", transport)
    if hasattr(module, "_exchange"):
        monkeypatch.setattr(module, "_exchange", transport)
        monkeypatch.setattr(module.asyncio, "sleep", AsyncMock())
    args, kwargs = _arguments(function)
    sig = inspect.signature(function)
    bound = sig.bind(*args, billing=Billing.NO_COMMERCE, **kwargs)
    if function.__name__ == "cloud_client":
        bound.arguments["api_key"] = "fake"
    if "client" in bound.arguments:
        bound.arguments["client"] = _FreeClient()
    if "params" in bound.arguments and function is proxies.connect_device:
        bound.arguments["params"] = {}
    if function is apps.upload_app:
        path = tmp_path / "app.apk"
        path.write_bytes(b"fake APK")
        bound.arguments["path"] = path
        monkeypatch.setattr(
            apps, "inspect_apk", lambda path: ("example", 1, "1", "Example")
        )
    if function is files.upload_file:
        path = tmp_path / "sample.txt"
        path.write_bytes(b"sample")
        bound.arguments["path"] = path
    if function.__name__ == "poll_token":
        bound.arguments["code"] = NS(interval=1, expires_in=10, device_code="fake")

    async def exercise():
        result = function(*bound.args, **bound.kwargs)
        if inspect.isasyncgen(result):
            await anext(result)
        elif inspect.isawaitable(result):
            await result

    with pytest.raises(_TransportReached):
        asyncio.run(exercise())


def test_free_boundaries_accept_no_commerce():
    retrieve = AsyncMock(return_value="result")
    client = NS(
        devices=NS(retrieve=retrieve, retrieve_capabilities=retrieve),
        numbers=NS(retrieve=retrieve),
        apps=NS(retrieve=retrieve),
        credentials=NS(packages=NS(list=AsyncMock(return_value=NS(data=[])))),
    )

    async def run():
        assert (
            await devices.capabilities("id", client=client, billing=Billing.NO_COMMERCE)
            == "result"
        )
        assert (
            await devices.retrieve("id", client=client, billing=Billing.NO_COMMERCE)
            == "result"
        )
        assert (
            await numbers.retrieve("id", client=client, billing=Billing.NO_COMMERCE)
            == "result"
        )
        assert await apps.get_app(client, "id", billing=Billing.NO_COMMERCE) == "result"
        assert (
            await secrets.list_package(client, "package", billing=Billing.NO_COMMERCE)
            == []
        )

    asyncio.run(run())


def test_core_objects_snapshot_settings_billing(monkeypatch):
    from mobilerun.core.credentials import CloudCredentials
    from mobilerun.core.devices.cloud import CloudDevice
    from mobilerun.core.mailbox import CloudMailbox as Mailbox
    from mobilerun.core.operator.cloud import CloudTaskOperator
    from mobilerun.core.phone_number import CloudPhoneNumber as PhoneNumber

    monkeypatch.setenv("MOBILERUN_BILLING", "paid")
    assert get_settings().MOBILERUN_BILLING is Billing.PAID
    device = CloudDevice("existing", api_key="fake")
    instances = [
        device,
        CloudTaskOperator(),
        PhoneNumber("existing"),
        Mailbox("existing"),
        CloudCredentials(api_key="fake"),
    ]
    monkeypatch.setenv("MOBILERUN_BILLING", "no_commerce")
    assert all(instance._billing is Billing.PAID for instance in instances)
