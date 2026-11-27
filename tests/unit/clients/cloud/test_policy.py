"""Cloud commerce policy, offline SDK transport, and public error contracts."""

import ast
import asyncio
import importlib
import inspect
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS
from typing import cast
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncMobilerun,
)

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
from mobilerun.clients.local import login_store as cloud_credentials
from mobilerun.config.settings import get_settings
from mobilerun.errors import (
    BillingError,
    CloudAPIError,
    CloudAuthenticationError,
    CloudConflict,
    CloudConnectionError,
    CloudPermissionDenied,
    CloudRateLimited,
    CloudResourceNotFound,
    CloudServiceUnavailable,
    CloudTimeout,
    CloudValidationError,
    CredentialFormatError,
    CredentialReadError,
)
from mobilerun.models.billing import Billing
from tests.support.paths import REPO_ROOT


def _boundaries():
    root = REPO_ROOT / "mobilerun/clients/cloud"
    for path in sorted(root.glob("*.py")):
        if path.stem in {"billing", "checkout", "errors", "payments"}:
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
    with pytest.raises(
        BillingError, match="Buy Connect proxy.*requires billing='paid'"
    ):
        proxies.buy_proxy("US", billing=mode)


class _TransportReached(BaseException):
    pass


class _FreeClient:
    def __getattr__(self, name):
        raise _TransportReached


@pytest.mark.parametrize("function", list(_boundaries()))
def test_no_commerce_boundary_policy(function, monkeypatch, tmp_path):
    module = importlib.import_module(function.__module__)
    if function in (
        devices.create,
        numbers.create,
        numbers.send,
        mailboxes.create,
        tasks.run,
        tasks.run_task,
        proxies.buy_proxy,
    ):
        args, kwargs = _arguments(function)
        with pytest.raises(BillingError, match="requires billing="):
            function(*args, billing=Billing.NO_COMMERCE, **kwargs)
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
    if function is secrets.save:
        bound.arguments["fields"] = {"username": "test"}
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


@pytest.mark.parametrize(
    "resource",
    ["cloud phones", "numbers", "mailboxes", "Connect proxies"],
)
@pytest.mark.parametrize("setting", list(Billing))
@pytest.mark.parametrize(
    "explicit", [None, *Billing, "paid", "included", "no_commerce"]
)
def test_policy_matrix(monkeypatch, resource, setting, explicit):
    monkeypatch.setenv("MOBILERUN_BILLING", setting)
    effective = None if explicit is None else Billing(explicit)
    blocked = effective in (None, Billing.NO_COMMERCE) or (
        effective is Billing.INCLUDED and resource == "Connect proxies"
    )
    if resource == "Connect proxies":
        # Stop at the client factory, even when the purchase is authorized.
        def client(*args, **kwargs):
            raise _TransportReached

        monkeypatch.setattr(proxies, "cloud_client", client)
        if not blocked:
            with pytest.raises(_TransportReached):
                asyncio.run(proxies.buy_proxy("US", billing=explicit))
            return

        def call():
            return proxies.buy_proxy("US", billing=explicit)

    else:
        adapter = {"cloud phones": devices, "numbers": numbers, "mailboxes": mailboxes}[
            resource
        ]
        endpoint = AsyncMock(
            return_value=NS(
                id="created",
                data=NS(
                    number_id="created",
                    id="created",
                    state="active",
                    status="active",
                    checkout_url=None,
                ),
            )
        )
        client = NS(
            **{
                "devices" if resource == "cloud phones" else resource: NS(
                    create=endpoint
                )
            }
        )
        params = (
            {"proxy": {"connect": {"id": "existing"}}}
            if resource == "cloud phones"
            else {}
        )

        def call():
            return adapter.create(billing=explicit, client=client, **params)

    if blocked:
        with pytest.raises(BillingError, match="given.*requires.*MOBILERUN_BILLING"):
            asyncio.run(call())
        if resource != "Connect proxies":
            endpoint.assert_not_called()
    else:
        expected = (
            ("included_only" if effective is Billing.INCLUDED else "included")
            if resource in {"numbers", "mailboxes"}
            else ("subscription" if effective is Billing.INCLUDED else "auto")
        )
        asyncio.run(call())
        key = "billing" if resource == "cloud phones" else "billing_preference"
        assert endpoint.call_args.kwargs[key] == expected


@pytest.mark.parametrize(
    "adapter,resource",
    [(numbers, "numbers"), (mailboxes, "mailboxes"), (devices, "devices")],
)
@pytest.mark.parametrize("mode", list(Billing))
def test_resource_calls_cannot_skip_policy(adapter, resource, mode):
    create = AsyncMock(
        return_value=NS(
            id="created",
            data=NS(
                number_id="created",
                id="created",
                state="active",
                status="active",
                checkout_url=None,
            ),
        )
    )
    client = NS(**{resource: NS(create=create)})
    params = {"proxy": {"connect": {"id": "existing"}}} if resource == "devices" else {}
    if mode is Billing.NO_COMMERCE:
        with pytest.raises(ValueError):
            asyncio.run(adapter.create(client=client, billing=mode, **params))
        create.assert_not_awaited()
    else:
        asyncio.run(adapter.create(client=client, billing=mode, **params))
        expected = (
            ("subscription" if mode is Billing.INCLUDED else "auto")
            if resource == "devices"
            else ("included_only" if mode is Billing.INCLUDED else "included")
        )
        create.assert_awaited_once_with(
            **{"billing" if resource == "devices" else "billing_preference": expected},
            **params,
        )


@pytest.mark.parametrize("mode", ["no_commerce", "included", "paid"])
@pytest.mark.parametrize("plan", [None, "auto", "minute", "subscription"])
def test_cloud_phone_plan_policy(mode, plan):
    blocked = mode == "no_commerce" or (
        mode == "included" and plan in {"auto", "minute"}
    )
    if blocked:
        with pytest.raises(ValueError):
            asyncio.run(devices.create(billing=mode, plan=plan))
    else:
        expected = plan or ("subscription" if mode == "included" else "auto")
        create = AsyncMock()
        asyncio.run(
            devices.create(
                billing=mode,
                plan=plan,
                proxy={"connect": {"id": "existing"}},
                client=NS(devices=NS(create=create)),
            )
        )
        assert create.call_args.kwargs["billing"] == expected


@pytest.mark.parametrize(
    "adapter,resource",
    [(numbers, "numbers"), (mailboxes, "mailboxes"), (devices, "devices")],
)
@pytest.mark.parametrize(
    "field", ["billing", "billingPreference", "billing_preference", "proxy"]
)
def test_extra_body_cannot_override_commerce_policy(adapter, resource, field):
    create = AsyncMock()
    client = NS(**{resource: NS(create=create)})
    with pytest.raises(ValueError):
        asyncio.run(
            adapter.create(
                client=client, billing="included", extra_body={field: "rent"}
            )
        )
    create.assert_not_awaited()


def _offline_sdk(handler):
    return AsyncMobilerun(
        api_key="test",
        base_url="https://sdk.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.parametrize("adapter", [devices, numbers, mailboxes])
@pytest.mark.parametrize(
    "field", ["billing", "billingPreference", "billing_preference", "proxy"]
)
def test_extra_body_cannot_reach_transport(adapter, field):
    seen = []

    async def run():
        async with _offline_sdk(
            lambda request: seen.append(request) or httpx.Response(400)
        ) as client:
            with pytest.raises(ValueError):
                await adapter.create(
                    client=client, billing="included", extra_body={field: "rent"}
                )

    asyncio.run(run())
    assert not seen


@pytest.mark.parametrize(
    "mode,plan,expected",
    [
        ("paid", None, "auto"),
        ("paid", "auto", "auto"),
        ("paid", "minute", "minute"),
        ("paid", "subscription", "subscription"),
        ("included", None, "subscription"),
        ("included", "subscription", "subscription"),
    ],
)
def test_plan_reaches_real_sdk_body(mode, plan, expected):
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(400, json={"message": "test rejection"})

    async def run():
        async with _offline_sdk(handler) as client:
            with pytest.raises(CloudAPIError):
                await devices.create(
                    client=client,
                    billing=mode,
                    plan=plan,
                    device_type="android_cloud_phone",
                    proxy={"connect": {"id": "existing"}},
                )

    asyncio.run(run())
    assert len(seen) == 1
    assert seen[0]["billing"] == expected


@pytest.mark.parametrize(
    "mode,plan",
    [
        ("no_commerce", None),
        ("no_commerce", "auto"),
        ("no_commerce", "minute"),
        ("no_commerce", "subscription"),
        ("included", "auto"),
        ("included", "minute"),
    ],
)
def test_blocked_plan_never_reaches_transport(mode, plan):
    seen = []

    async def run():
        async with _offline_sdk(
            lambda request: seen.append(request) or httpx.Response(400)
        ) as client:
            with pytest.raises(ValueError):
                await devices.create(
                    client=client,
                    billing=mode,
                    plan=plan,
                    proxy={"connect": {"id": "existing"}},
                )

    asyncio.run(run())
    assert not seen


def test_extra_query_cannot_override_device_billing():
    seen = []

    async def run():
        async with _offline_sdk(
            lambda request: seen.append(request) or httpx.Response(400)
        ) as client:
            with pytest.raises(ValueError):
                await devices.create(
                    client=client,
                    billing=Billing.INCLUDED,
                    extra_query={"billing": "minute"},
                )

    asyncio.run(run())
    assert not seen


@pytest.mark.parametrize(
    "proxy",
    [
        {},
        {"connect": {}},
        {"connect": {"country": None}},
        {"socks5": {"host": "example.test"}},
    ],
)
def test_invalid_explicit_cloud_phone_proxy_is_rejected_before_http(proxy):
    seen = []

    async def run():
        async with _offline_sdk(
            lambda request: seen.append(request) or httpx.Response(400)
        ) as client:
            with pytest.raises(ValueError, match="proxy"):
                await devices.create(client=client, billing=Billing.PAID, proxy=proxy)

    asyncio.run(run())
    assert not seen


@pytest.mark.parametrize(
    "billing,proxy,allowed",
    [
        ("included", {"connect": {"country": "us"}}, False),
        ("included", {"connect": {"id": "existing"}}, True),
        ("included", {"socks5": {"host": "example.test", "port": 1080}}, True),
        ("paid", {"connect": {"country": "us"}}, True),
        ("no_commerce", {"connect": {"id": "existing"}}, False),
    ],
)
def test_proxy_billing_is_checked_before_transport(billing, proxy, allowed):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(400, json={"message": "test rejection"})

    async def run():
        async with _offline_sdk(handler) as client:
            with pytest.raises(CloudAPIError):
                await devices.create(client=client, billing=billing, proxy=proxy)

    asyncio.run(run())
    assert bool(seen) is allowed


@pytest.mark.parametrize("adapter", [numbers, mailboxes])
@pytest.mark.parametrize(
    "mode,expected", [("included", "included_only"), ("paid", "included")]
)
def test_number_and_mailbox_billing_serialized(adapter, mode, expected):
    import json

    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(400, json={"message": "test rejection"})

    async def run():
        async with _offline_sdk(handler) as client:
            with pytest.raises(CloudAPIError):
                await adapter.create(client=client, billing=mode)

    asyncio.run(run())
    assert bodies[0]["billingPreference"] == expected


@pytest.mark.parametrize("options", [{}, {"proxy": None}])
def test_missing_proxy_is_rejected_before_transport(options):
    seen = []

    def handler(request):
        import json

        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "phone", "name": None})

    async def run():
        async with _offline_sdk(handler) as client:
            with pytest.raises(ValueError, match="proxy"):
                await devices.create(client=client, billing=Billing.PAID, **options)

    asyncio.run(run())
    assert not seen


@pytest.mark.parametrize("resource", ["device", "number", "mailbox", "connect", "buy"])
@pytest.mark.parametrize("extra", ["extra_body", "extra_query"])
@pytest.mark.parametrize(
    "field", ["proxy", "connect", "billing", "billingPreference", "billingMode"]
)
def test_commerce_fields_in_extras_never_reach_real_transport(
    monkeypatch, resource, extra, field
):
    from contextlib import nullcontext

    from mobilerun.clients.cloud import proxies

    seen = []

    async def run():
        async with _offline_sdk(
            lambda request: seen.append(request) or httpx.Response(400)
        ) as client:
            monkeypatch.setattr(
                proxies, "cloud_client", lambda *args: nullcontext(client)
            )
            extras = {extra: {field: {"country": "DE"}}}
            # Deliberately send invalid fields to exercise runtime commerce guards.
            with pytest.raises(ValueError):
                if resource == "device":
                    await cast(Callable[..., Awaitable[object]], devices.create)(
                        client=client,
                        billing="paid",
                        proxy={"connect": {"id": "existing"}},
                        **extras,
                    )
                elif resource == "number":
                    await cast(Callable[..., Awaitable[object]], numbers.create)(
                        client=client, billing=Billing.PAID, **extras
                    )
                elif resource == "mailbox":
                    await cast(Callable[..., Awaitable[object]], mailboxes.create)(
                        client=client, billing=Billing.PAID, **extras
                    )
                elif resource == "connect":
                    await cast(
                        Callable[..., Awaitable[object]], proxies.connect_device
                    )("phone", extras, billing=Billing.NO_COMMERCE)
                else:
                    await cast(Callable[..., Awaitable[object]], proxies.buy_proxy)(
                        "DE", billing=Billing.PAID, **extras
                    )

    asyncio.run(run())
    assert not seen


def test_cloud_helpers_do_not_import_sdk():
    """Checkout and error helpers are not independent transport boundaries."""
    for name in ("checkout", "errors", "payments"):
        path = REPO_ROOT / "mobilerun/clients/cloud" / f"{name}.py"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("mobilerun_sdk") for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("mobilerun_sdk")


def test_cloud_modules_require_explicit_billing():
    for path in (REPO_ROOT / "mobilerun/clients/cloud").glob("*.py"):
        if path.stem in {"billing", "checkout", "errors", "payments"}:
            continue  # Helpers do not expose billable cloud operations.
        tree = ast.parse(path.read_text())
        assert not any(
            isinstance(node, ast.ClassDef)
            and not (path.stem == "device_flow" and node.name == "CodeResponse")
            for node in ast.walk(tree)
        ), path
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_") or node.name == "requires":
                continue
            assert any(
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Name)
                and dec.func.id == "requires"
                for dec in node.decorator_list
            ), (path, node.name)
            names = [argument.arg for argument in node.args.kwonlyargs]
            assert "billing" in names, (path, node.name)
            assert node.args.kw_defaults[names.index("billing")] is None, (
                path,
                node.name,
            )
        assert "mobilerun.core" not in path.read_text(), path


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, CloudAuthenticationError),
        (403, CloudPermissionDenied),
        (404, CloudResourceNotFound),
        (409, CloudConflict),
        (400, CloudValidationError),
        (422, CloudValidationError),
        (429, CloudRateLimited),
        (500, CloudServiceUnavailable),
        (503, CloudServiceUnavailable),
    ],
)
@pytest.mark.parametrize(
    "boundary", ["number", "mailbox", "vault", "task", "app", "device"]
)
def test_sdk_errors_are_typed_sanitized_and_chained(status, expected, boundary):
    response = httpx.Response(
        status,
        headers={"x-request-id": "req-safe"},
        request=httpx.Request("GET", "https://private.test/signed?token=secret"),
    )
    error = APIStatusError(
        "secret request body", response=response, body={"password": "secret"}
    )
    endpoint = AsyncMock(side_effect=error)
    if boundary == "number":
        coroutine = numbers.retrieve(
            "id", billing="included", client=NS(numbers=NS(retrieve=endpoint))
        )
    elif boundary == "mailbox":
        coroutine = mailboxes.retrieve(
            "id", billing="included", client=NS(mailboxes=NS(retrieve=endpoint))
        )
    elif boundary == "vault":
        coroutine = secrets.list_package(
            NS(credentials=NS(packages=NS(list=endpoint))),
            "package",
            billing="included",
        )
    elif boundary == "task":
        coroutine = tasks.status(
            NS(tasks=NS(get_status=endpoint)), "id", billing="included"
        )
    elif boundary == "app":
        coroutine = apps.get_app(
            NS(apps=NS(retrieve=endpoint)), "id", billing="included"
        )
    else:
        coroutine = devices.retrieve(
            "id", billing="included", client=NS(devices=NS(retrieve=endpoint))
        )
    with pytest.raises(expected) as raised:
        asyncio.run(coroutine)
    assert isinstance(raised.value, CloudAPIError)
    assert raised.value.__cause__ is error
    message = str(raised.value)
    assert f"HTTP {status}" in message and "req-safe" in message
    assert "secret" not in message and "private.test" not in message


@pytest.mark.parametrize(
    "error_type,expected",
    [(APIConnectionError, CloudConnectionError), (APITimeoutError, CloudTimeout)],
)
def test_transport_errors(error_type, expected):
    error = error_type(request=httpx.Request("GET", "https://private.test"))
    client = NS(tasks=NS(get_status=AsyncMock(side_effect=error)))
    with pytest.raises(expected) as raised:
        asyncio.run(tasks.status(client, "id", billing="included"))
    assert raised.value.__cause__ is error


@pytest.mark.parametrize("content", ["{broken", "[]", '{"api_key": 123}'])
def test_corrupt_credentials_are_not_silent_logout(monkeypatch, tmp_path, content):
    path = tmp_path / "credentials.json"
    path.write_text(content)
    monkeypatch.setattr(cloud_credentials, "credential_path", lambda: path)
    with pytest.raises(CredentialFormatError):
        cloud_credentials.load_credentials()


def test_unreadable_credentials_are_not_silent_logout(monkeypatch, tmp_path):
    monkeypatch.setattr(cloud_credentials, "credential_path", lambda: tmp_path)
    with pytest.raises(CredentialReadError):
        cloud_credentials.load_credentials()


def test_missing_credentials_are_logged_out(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cloud_credentials, "credential_path", lambda: tmp_path / "missing"
    )
    assert cloud_credentials.load_credentials() is None


def test_stream_failure_is_sanitized_and_chained():
    error = httpx.ReadError("signed URL containing private token")

    class Stream:
        async def iter_lines(self):
            yield "data: {}"
            raise error

    @asynccontextmanager
    async def attach(*args, **kwargs):
        yield Stream()

    client = NS(tasks=NS(with_streaming_response=NS(attach=attach)))

    async def consume():
        return [line async for line in tasks.progress(client, "id", billing="included")]

    with pytest.raises(CloudConnectionError) as raised:
        asyncio.run(consume())
    assert "private" not in str(raised.value)
    assert raised.value.__cause__ is error


def test_auth_response_body_never_reaches_public_errors(monkeypatch):
    from mobilerun.clients.cloud import device_flow

    response = httpx.Response(
        403,
        text="private token and credentials",
        request=httpx.Request("POST", "https://auth.test/device/code"),
    )
    monkeypatch.setattr(device_flow, "_request", AsyncMock(return_value=response))
    with pytest.raises(CloudPermissionDenied) as raised:
        asyncio.run(
            device_flow.request_code("https://auth.test", "client", billing="included")
        )
    assert "private" not in str(raised.value)
    assert raised.value.__cause__ is not None


def test_driver_action_sdk_error_is_sanitized():
    from mobilerun.core.devices.cloud import CloudDevice

    device = CloudDevice("id", api_key="fake")
    error = APIStatusError(
        "private response text",
        response=httpx.Response(
            401, request=httpx.Request("POST", "https://private.test")
        ),
        body=None,
    )
    device._write_client = NS(devices=NS(actions=NS(tap=AsyncMock(side_effect=error))))
    with pytest.raises(CloudAuthenticationError) as raised:
        asyncio.run(device.tap(1, 2))
    assert raised.value.__cause__ is error
    assert "private" not in str(raised.value)
