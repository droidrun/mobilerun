"""Cloud functions are explicit, billable-marked, and sanitize SDK failures."""

import ast
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import APIConnectionError, APIStatusError, APITimeoutError

from mobilerun.clients.cloud import apps, devices, mailboxes, numbers, secrets, tasks
from mobilerun.clients.local import login_store as cloud_credentials
from mobilerun.errors import (
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

ROOT = Path(__file__).resolve().parents[2]


def test_cloud_helpers_do_not_import_sdk():
    """Checkout and error helpers are not independent transport boundaries."""
    for name in ("checkout", "errors", "payments"):
        path = ROOT / "mobilerun/clients/cloud" / f"{name}.py"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("mobilerun_sdk") for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("mobilerun_sdk")


def test_cloud_modules_require_explicit_billing():
    for path in (ROOT / "mobilerun/clients/cloud").glob("*.py"):
        if path.stem in {"checkout", "errors", "payments"}:
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
