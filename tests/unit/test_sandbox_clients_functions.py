"""Offline tests for the guarded cloud function boundary."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from pydantic import BaseModel, Field

from mobilerun.errors.cloud import BillingError
from mobilerun.models.billing import Billing
from mobilerun.sandbox import clients as functions


def run(coro):
    return asyncio.run(coro)


def client():
    versions = SimpleNamespace(
        create=AsyncMock(), retrieve=AsyncMock(), list=AsyncMock()
    )
    funcs = SimpleNamespace(
        create=AsyncMock(),
        source=AsyncMock(),
        delete=AsyncMock(),
        list=AsyncMock(),
        versions=versions,
    )
    logs = AsyncMock()
    sdk = SimpleNamespace(
        compute=SimpleNamespace(functions=funcs, runs=SimpleNamespace(logs=logs))
    )
    return sdk


class Owned:
    def __init__(self, sdk):
        self.sdk = sdk
        self.closed = False

    async def __aenter__(self):
        return self.sdk

    async def __aexit__(self, *args):
        self.closed = True


class Item(BaseModel):
    latest_version: int = Field(alias="latestVersion")


class Envelope(BaseModel):
    data: dict


def test_list_functions_paginates_models_and_empty_page():
    sdk = client()
    sdk.compute.functions.list.side_effect = [
        Envelope(
            data={"items": [Item(latestVersion=3)], "pagination": {"hasNext": True}}
        ),
        {"data": {"items": [], "pagination": {"hasNext": True}}},
        {"data": {"items": [{"id": "two"}], "pagination": {"hasNext": False}}},
    ]
    assert run(functions.list_functions(billing=Billing.NO_COMMERCE, client=sdk)) == [
        {"latestVersion": 3},
        {"id": "two"},
    ]
    assert sdk.compute.functions.list.call_count == 3


def test_list_versions_paginates_and_omits_null_items():
    sdk = client()
    sdk.compute.functions.versions.list.side_effect = [
        {"data": {"items": [None, {"version": 1}], "pagination": {"hasNext": True}}},
        {"data": {"items": [{"version": 2}], "pagination": {"hasNext": False}}},
    ]
    assert run(
        functions.list_versions("f", billing=Billing.NO_COMMERCE, client=sdk)
    ) == [{"version": 1}, {"version": 2}]
    sdk.compute.functions.versions.list.assert_any_await("f", page=2, page_size=100)


@pytest.mark.parametrize(
    "name,args,path,expected_call",
    [
        ("get_version", ("f", 2), "versions.retrieve", ((2,), {"function_id": "f"})),
        (
            "create_function",
            ("n", "src", "key"),
            "create",
            ((), {"name": "n", "source": "src", "idempotency_key": "key"}),
        ),
        (
            "create_version",
            ("f", "src", "key"),
            "versions.create",
            (("f",), {"source": "src", "idempotency_key": "key"}),
        ),
        ("delete_function", ("f",), "delete", (("f",), {})),
    ],
)
def test_sdk_dispatch(name, args, path, expected_call):
    sdk = client()
    method = sdk.compute.functions
    for part in path.split("."):
        method = getattr(method, part)
    method.return_value = {"data": {"id": "x"}}
    result = run(getattr(functions, name)(*args, billing=Billing.PAID, client=sdk))
    assert result == (None if name == "delete_function" else {"id": "x"})
    method.assert_awaited_once_with(*expected_call[0], **expected_call[1])


def test_get_source():
    sdk = client()
    sdk.compute.functions.source.return_value = {"data": {"source": "hello"}}
    assert (
        run(functions.get_source("f", 7, billing=Billing.NO_COMMERCE, client=sdk))
        == "hello"
    )
    sdk.compute.functions.source.assert_awaited_once_with("f", version=7)


@pytest.mark.parametrize(
    "name,args",
    [
        ("list_functions", ()),
        ("list_versions", ("f",)),
        ("get_version", ("f", 1)),
        ("get_source", ("f", 1)),
        ("create_function", ("n", "s", "k")),
        ("create_version", ("f", "s", "k")),
        ("delete_function", ("f",)),
        ("run_logs", ("r",)),
    ],
)
@pytest.mark.parametrize("billing", [None, "invalid"])
def test_invalid_billing_rejected_before_factory(monkeypatch, name, args, billing):
    factory = Mock()
    monkeypatch.setattr(functions, "cloud_client", factory)
    with pytest.raises(BillingError):
        run(getattr(functions, name)(*args, billing=billing))
    factory.assert_not_called()


@pytest.mark.parametrize(
    "name,args",
    [("create_function", ("n", "s", "k")), ("create_version", ("f", "s", "k"))],
)
@pytest.mark.parametrize("billing", [Billing.NO_COMMERCE, Billing.INCLUDED])
def test_paid_creations_reject_lower_modes_before_factory(
    monkeypatch, name, args, billing
):
    factory = Mock()
    monkeypatch.setattr(functions, "cloud_client", factory)
    with pytest.raises(BillingError):
        run(getattr(functions, name)(*args, billing=billing))
    factory.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_owned_client_closed_and_transport_errors_sanitized(monkeypatch, fails):
    sdk = client()
    sdk.compute.functions.delete.return_value = None
    if fails:
        sdk.compute.functions.delete.side_effect = httpx.ConnectError(
            "secret-url", request=httpx.Request("GET", "https://secret.test/token")
        )
    owned = Owned(sdk)
    monkeypatch.setattr(functions, "cloud_client", Mock(return_value=owned))
    if fails:
        with pytest.raises(Exception) as caught:
            run(functions.delete_function("f", billing=Billing.NO_COMMERCE))
        assert "secret" not in str(caught.value)
        assert caught.value.__suppress_context__
    else:
        run(functions.delete_function("f", billing=Billing.NO_COMMERCE))
    assert owned.closed


def test_injected_client_not_closed():
    sdk = client()
    sdk.close = AsyncMock()
    run(functions.delete_function("f", billing=Billing.NO_COMMERCE, client=sdk))
    sdk.close.assert_not_awaited()


def test_run_logs_uses_separate_unauthenticated_client(monkeypatch):
    sdk = client()
    sdk.compute.runs.logs.return_value = {
        "data": {"url": "https://signed.example/log?secret=token"}
    }
    seen = []
    transport = httpx.MockTransport(
        lambda request: (seen.append(request), httpx.Response(200, text="logs"))[1]
    )
    real = httpx.AsyncClient
    monkeypatch.setattr(
        functions.httpx,
        "AsyncClient",
        lambda **kwargs: real(transport=transport, **kwargs),
    )
    assert (
        run(functions.run_logs("r", billing=Billing.NO_COMMERCE, client=sdk)) == "logs"
    )
    assert len(seen) == 1 and "authorization" not in seen[0].headers


def test_run_logs_http_error_does_not_expose_signed_url(monkeypatch):
    sdk = client()
    sdk.compute.runs.logs.return_value = {
        "data": {"url": "https://signed.example/log?secret=token"}
    }
    transport = httpx.MockTransport(
        lambda request: httpx.Response(403, request=request)
    )
    real = httpx.AsyncClient
    monkeypatch.setattr(
        functions.httpx,
        "AsyncClient",
        lambda **kwargs: real(transport=transport, **kwargs),
    )
    with pytest.raises(Exception) as caught:
        run(functions.run_logs("r", billing=Billing.NO_COMMERCE, client=sdk))
    assert "secret" not in str(caught.value) and "signed.example" not in str(
        caught.value
    )
    assert caught.value.__suppress_context__


@pytest.mark.parametrize(
    "url",
    [
        "https://host:secret-token/file",
        {"secret-token": "malformed"},
        "ftp://host/secret-token",
    ],
)
def test_run_logs_invalid_url_never_exposes_secret(url):
    import traceback

    sdk = client()
    sdk.compute.runs.logs.return_value = {"data": {"url": url}}
    with pytest.raises(Exception) as caught:
        run(functions.run_logs("r", billing=Billing.NO_COMMERCE, client=sdk))
    assert "secret-token" not in str(caught.value)
    assert "secret-token" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("error_type", [httpx.CookieConflict, httpx.StreamError])
def test_run_logs_other_httpx_errors_are_sanitized(monkeypatch, error_type):
    import traceback

    sdk = client()
    sdk.compute.runs.logs.return_value = {"data": {"url": "https://example.test"}}
    transport = httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(error_type("secret-token"))
    )
    real = httpx.AsyncClient
    monkeypatch.setattr(
        functions.httpx,
        "AsyncClient",
        lambda **kwargs: real(transport=transport, **kwargs),
    )
    with pytest.raises(Exception) as caught:
        run(functions.run_logs("r", billing=Billing.NO_COMMERCE, client=sdk))
    assert "secret-token" not in str(caught.value)
    assert "secret-token" not in "".join(traceback.format_exception(caught.value))


READ_OPERATIONS = [
    ("list_functions", (), "list"),
    ("list_versions", ("f",), "versions.list"),
    ("get_version", ("f", 1), "versions.retrieve"),
    ("get_source", ("f", 1), "source"),
    ("delete_function", ("f",), "delete"),
]
ALL_OPERATIONS = READ_OPERATIONS + [
    ("create_function", ("n", "s", "k"), "create"),
    ("create_version", ("f", "s", "k"), "versions.create"),
    ("run_logs", ("r",), "runs.logs"),
]


def sdk_method(sdk, path):
    target = sdk.compute if path.startswith("runs.") else sdk.compute.functions
    for part in path.split("."):
        target = getattr(target, part)
    return target


@pytest.mark.parametrize("name,args,path", ALL_OPERATIONS)
def test_billing_required_before_factory(monkeypatch, name, args, path):
    factory = Mock()
    monkeypatch.setattr(functions, "cloud_client", factory)
    with pytest.raises(TypeError):
        getattr(functions, name)(*args)
    factory.assert_not_called()


@pytest.mark.parametrize("name,args,path", READ_OPERATIONS)
@pytest.mark.parametrize("billing", list(Billing))
def test_read_and_delete_accept_all_billing(name, args, path, billing):
    sdk = client()
    method = sdk_method(sdk, path)
    method.return_value = {
        "data": {"items": [], "pagination": {"hasNext": False}, "source": "source"}
    }
    run(getattr(functions, name)(*args, billing=billing, client=sdk))
    method.assert_awaited_once()


@pytest.mark.parametrize("name,args,path", ALL_OPERATIONS)
def test_sdk_errors_are_sanitized_for_every_operation(name, args, path):
    sdk = client()
    sdk.close = AsyncMock()
    sdk_method(sdk, path).side_effect = httpx.ConnectError(
        "secret-url", request=httpx.Request("GET", "https://secret.test/token")
    )
    with pytest.raises(Exception) as caught:
        run(getattr(functions, name)(*args, billing=Billing.PAID, client=sdk))
    assert "secret" not in str(caught.value)
    assert caught.value.__suppress_context__
    sdk.close.assert_not_awaited()


def test_sdk_transport_uses_generated_endpoints_and_idempotency():
    from mobilerun_sdk import AsyncMobilerun

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": "id",
                    "source": "hello",
                    "items": [],
                    "pagination": {"hasNext": False},
                }
            },
        )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with AsyncMobilerun(
                api_key="fake", base_url="https://example.test/v1", http_client=http
            ) as sdk:
                await functions.list_functions(billing=Billing.NO_COMMERCE, client=sdk)
                await functions.list_versions(
                    "f", billing=Billing.NO_COMMERCE, client=sdk
                )
                await functions.get_version(
                    "f", 2, billing=Billing.NO_COMMERCE, client=sdk
                )
                await functions.get_source(
                    "f", 2, billing=Billing.NO_COMMERCE, client=sdk
                )
                await functions.create_function(
                    "n", "src", "key1", billing=Billing.PAID, client=sdk
                )
                await functions.create_version(
                    "f", "src", "key2", billing=Billing.PAID, client=sdk
                )
                await functions.delete_function(
                    "f", billing=Billing.NO_COMMERCE, client=sdk
                )

    run(exercise())
    assert [r.url.path for r in requests] == [
        "/v1/compute/functions",
        "/v1/compute/functions/f/versions",
        "/v1/compute/functions/f/versions/2",
        "/v1/compute/functions/f/source",
        "/v1/compute/functions",
        "/v1/compute/functions/f/versions",
        "/v1/compute/functions/f",
    ]
    assert requests[4].headers["idempotency-key"] == "key1"
    assert requests[5].headers["idempotency-key"] == "key2"


@pytest.mark.parametrize("billing", list(Billing))
@pytest.mark.parametrize("fail_download", [False, True])
def test_logs_owns_both_clients_and_accepts_all_billing(
    monkeypatch, billing, fail_download
):
    sdk = client()
    sdk.compute.runs.logs.return_value = {
        "data": {"url": "https://signed.example/log?token=secret"}
    }
    owned = Owned(sdk)
    monkeypatch.setattr(functions, "cloud_client", Mock(return_value=owned))
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(500 if fail_download else 200, text="ok")

    transport = httpx.MockTransport(respond)
    actual = httpx.AsyncClient
    http_clients = []

    def make_client(**kwargs):
        downloader = actual(transport=transport, **kwargs)
        http_clients.append(downloader)
        return downloader

    monkeypatch.setattr(functions.httpx, "AsyncClient", make_client)
    if fail_download:
        import traceback

        with pytest.raises(Exception) as caught:
            run(functions.run_logs("r", billing=billing))
        formatted = "".join(traceback.format_exception(caught.value))
        assert "token=secret" not in formatted and "signed.example" not in formatted
    else:
        assert run(functions.run_logs("r", billing=billing)) == "ok"
    assert owned.closed and http_clients[0].is_closed
    assert "authorization" not in seen[0].headers


def test_logs_follow_redirect_without_credentials(monkeypatch):
    sdk = client()
    sdk.compute.runs.logs.return_value = {"data": {"url": "https://signed.example/a"}}
    requests = []

    def respond(request):
        requests.append(request)
        return (
            httpx.Response(302, headers={"location": "https://other.example/b"})
            if len(requests) == 1
            else httpx.Response(200, text="ok")
        )

    transport = httpx.MockTransport(respond)
    actual = httpx.AsyncClient
    monkeypatch.setattr(
        functions.httpx,
        "AsyncClient",
        lambda **kwargs: actual(transport=transport, **kwargs),
    )
    assert run(functions.run_logs("r", billing=Billing.NO_COMMERCE, client=sdk)) == "ok"
    assert len(requests) == 2
    assert all("authorization" not in request.headers for request in requests)
