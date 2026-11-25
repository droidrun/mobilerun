"""Signed transfer logs and vault failures must be safe at the direct API boundary."""

import asyncio
import logging
import traceback
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import APIStatusError

from mobilerun.clients.cloud import apps, files, secrets
from mobilerun.clients.cloud.errors import cloud_error
from mobilerun.core.credentials.cloud import CloudCredentials
from mobilerun.errors import CloudAPIError, CloudPermissionDenied, CloudValidationError
from mobilerun.sandbox.clients import run_logs


@pytest.mark.parametrize("kind", ["logs", "file", "app"])
@pytest.mark.parametrize("status", [200, 403])
def test_signed_transfers_hide_queries(kind, status, monkeypatch, caplog, tmp_path):
    url = "https://storage.invalid/blob?sig=first-canary"
    redirected = "https://cdn.invalid/log?sig=second-canary"
    seen = []

    def respond(request):
        seen.append(str(request.url))
        if kind == "logs" and len(seen) == 1:
            return httpx.Response(302, headers={"Location": redirected})
        return httpx.Response(status, text="data")

    caplog.set_level(logging.INFO, logger="httpx")
    actual = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: actual(transport=httpx.MockTransport(respond), **kw),
    )
    sdk = AsyncMock()
    kw = {"billing": "no_commerce"}
    if kind == "logs":
        sdk.compute.runs.logs.return_value = {"url": url}
        call = run_logs("id", client=sdk, **kw)
    else:
        path = tmp_path / "file.apk"
        path.write_bytes(b"data")
        if kind == "app":
            monkeypatch.setattr(apps, "inspect_apk", lambda _: ("pkg", 1, "1", "name"))
            monkeypatch.setattr(apps, "cloud_client", lambda *a, **kw: nullcontext(sdk))
            sdk.apps.create_signed_upload_url.return_value = NS(
                version_id="v",
                app_id="a",
                r2_upload_urls=[NS(file_name=path.name, r2_upload_url=url)],
            )
            call = apps.upload_app(path, **kw)
        else:
            sdk.files.upload_url.return_value = NS(put_url=url, file_id="file")
            sdk.files.confirm.return_value = NS(
                file_id="file",
                filename=path.name,
                mime_type="application/octet-stream",
                size_bytes=4,
                created_at=None,
            )
            call = files.upload_file(sdk, path, **kw)

    text = ""
    if status == 403:
        with pytest.raises(CloudAPIError) as err:
            asyncio.run(call)
        text = "".join(traceback.format_exception(err.value))
    else:
        asyncio.run(call)
    assert seen == ([url, redirected] if kind == "logs" else [url])
    for marker in ("first-canary", "second-canary"):
        assert marker not in text
        assert all(marker not in record.getMessage() for record in caplog.records)


def rejection(status, body):
    response = httpx.Response(
        status,
        headers={"x-request-id": "vault-req"},
        request=httpx.Request("POST", "https://vault.invalid"),
    )
    return APIStatusError(str(body), response=response, body=body)


@pytest.mark.parametrize("action", ["save", "create_field", "update"])
def test_vault_sdk_echo_not_in_traceback(action):
    sdk = AsyncMock()
    credential = sdk.credentials.packages.credentials
    password = "vault-canary"
    method, args = {
        "save": (credential.create, ({"password": password},)),
        "create_field": (credential.fields.create, ("password", password)),
        "update": (credential.fields.update, ("password", password)),
    }[action]
    method.side_effect = rejection(403, {"echo": password})
    with pytest.raises(CloudPermissionDenied) as err:
        asyncio.run(
            getattr(secrets, action)(sdk, "pkg", "name", *args, billing="no_commerce")
        )
    assert password not in "".join(traceback.format_exception(err.value))


def test_credentials_update_preserves_safe_diagnostics(monkeypatch):
    sdk = AsyncMock()
    sdk.credentials.packages.credentials.fields.update.side_effect = rejection(
        403, {"echo": "vault-canary"}
    )
    monkeypatch.setattr(
        "mobilerun.core.credentials.cloud.cloud_client", lambda *a, **kw: sdk
    )
    vault = CloudCredentials()
    with pytest.raises(CloudPermissionDenied) as err:
        asyncio.run(vault.update("pkg", "name", "password", "vault-canary"))
    assert (err.value.status, err.value.request_id, err.value.value_safe) == (
        403,
        "vault-req",
        True,
    )
    assert err.value.__context__ is None and err.value.__cause__ is None


def test_validation_with_null_location_is_still_a_cloud_error():
    exc = rejection(422, {"detail": [{"loc": None}]})
    assert type(cloud_error(exc, "Cloud request")) is CloudValidationError


def test_credentials_update_drops_unsafe_notes_and_causes(monkeypatch):
    error = CloudPermissionDenied(status=403, request_id="support-id")
    password = "secret-note-canary"
    error.add_note(password)
    error.__cause__ = RuntimeError(password)
    monkeypatch.setattr(secrets, "update", AsyncMock(side_effect=error))
    monkeypatch.setattr(
        "mobilerun.core.credentials.cloud.cloud_client", lambda *a, **kw: AsyncMock()
    )
    with pytest.raises(CloudPermissionDenied) as err:
        asyncio.run(CloudCredentials().update("pkg", "name", "password", password))
    assert err.value.request_id == "support-id"
    assert password not in "".join(traceback.format_exception(err.value))
