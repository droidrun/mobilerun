"""Secret redaction, signed transfer diagnostics, and cloud vault contracts."""

import asyncio
import logging
import re
import time
import traceback
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from mobilerun_sdk import APIStatusError, AsyncMobilerun, BadRequestError, NotFoundError

from mobilerun.clients.cloud import apps, devices, files, proxies, secrets
from mobilerun.clients.cloud.errors import cloud_error
from mobilerun.core.credentials.cloud import CloudCredentials
from mobilerun.errors import CloudAPIError, CloudPermissionDenied, CloudValidationError
from mobilerun.models.billing import Billing
from mobilerun.sandbox.clients import run_logs
from mobilerun.utils.redaction import _redact_secrets, secret_scope
from tests.support.devices import MemoryDevice


@pytest.mark.parametrize(
    "operation", ["device", "proxy", "save", "create_field", "update"]
)
def test_sdk_request_options_never_log_secrets(monkeypatch, caplog, operation):
    secret = "sdk-sentinel-password-987654"
    caplog.set_level(logging.DEBUG, logger="mobilerun_sdk._base_client")
    sdk = AsyncMobilerun(
        api_key="test",
        base_url="https://example.test",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(400, json={"message": "rejected"})
            )
        ),
    )

    @asynccontextmanager
    async def client(*args, **kwargs):
        yield sdk

    monkeypatch.setattr(proxies, "cloud_client", client)

    async def run():
        try:
            with pytest.raises(CloudValidationError):
                if operation == "device":
                    await devices.create(
                        client=sdk,
                        billing=Billing.PAID,
                        proxy={
                            "socks5": {
                                "host": "example.test",
                                "port": 1080,
                                "password": secret,
                            }
                        },
                    )
                elif operation == "proxy":
                    await proxies.connect_device(
                        "device",
                        {
                            "socks5": {
                                "host": "example.test",
                                "port": 1080,
                                "password": secret,
                            }
                        },
                        billing=Billing.NO_COMMERCE,
                    )
                elif operation == "save":
                    await secrets.save(
                        sdk,
                        "pkg",
                        "name",
                        {"password": secret},
                        billing=Billing.NO_COMMERCE,
                    )
                elif operation == "create_field":
                    await secrets.create_field(
                        sdk,
                        "pkg",
                        "name",
                        "password",
                        secret,
                        billing=Billing.NO_COMMERCE,
                    )
                else:
                    await secrets.update(
                        sdk,
                        "pkg",
                        "name",
                        "password",
                        secret,
                        billing=Billing.NO_COMMERCE,
                    )
        finally:
            await sdk.close()

    asyncio.run(run())
    assert any("Request options" in r.getMessage() for r in caplog.records)
    assert all(secret not in repr(r.__dict__) for r in caplog.records)


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


def test_vault_write_debug_logs_do_not_disclose_values(caplog):
    secret = "vault-write-secret-canary"
    log = logging.getLogger("mobilerun_sdk._base_client")
    caplog.set_level(logging.DEBUG, logger=log.name)

    async def update(client, package, name, field, value, **kwargs):
        log.debug("request %s", {"value": value})

    async def run():
        vault = CloudCredentials(api_key="dummy")
        with patch(
            "mobilerun.core.credentials.cloud.secrets.update",
            AsyncMock(side_effect=update),
        ):
            await vault.update("app", "work", "password", secret)
        await vault.close()

    asyncio.run(run())
    assert secret not in caplog.text


def test_vault_write_failure_redacts_secret():
    secret = "vault-error-secret-canary"

    async def run():
        vault = CloudCredentials(api_key="dummy")
        with patch(
            "mobilerun.core.credentials.cloud.secrets.update",
            AsyncMock(side_effect=RuntimeError(secret)),
        ):
            with pytest.raises(RuntimeError) as error:
                await vault.update("app", "work", "password", secret)
            assert secret not in str(error.value)
            assert error.value.__context__ is None
        await vault.close()

    asyncio.run(run())


class FakeVault:
    def __init__(self):
        self.data = {}
        self.credentials = self
        self.packages = self
        self.fields = Fields(self)

    async def list_all(self):
        return NS(data=[NS(package_name=p) for p in self.data])

    async def list(self, package_name=None, page=None):
        if page is not None:
            return NS(
                items=[
                    self.item(p, n, f)
                    for p, names in self.data.items()
                    for n, f in names.items()
                ],
                pagination=NS(has_next=False),
            )
        package_name = package_name.replace(".", "-")
        if package_name not in self.data:
            raise NotFoundError(
                "Package not found",
                response=httpx.Response(
                    404, request=httpx.Request("GET", "https://fake.invalid")
                ),
                body={"code": "credential_not_found"},
            )
        return NS(
            data=[
                self.item(package_name, n, f)
                for n, f in self.data[package_name].items()
            ]
        )

    def item(self, package, name, fields):
        return NS(
            package_name=package,
            credential_name=name,
            fields=[NS(field_type=k, value=v) for k, v in fields.items()],
        )

    async def create(self, package_name, credential_name=None, fields=None):
        package_name = package_name.replace(".", "-")
        if credential_name is None:
            if package_name in self.data:
                raise BadRequestError(
                    "Package exists",
                    response=httpx.Response(
                        400, request=httpx.Request("POST", "https://fake.invalid")
                    ),
                    body={"code": "package_exists"},
                )
            self.data[package_name] = {}
        else:
            self.data[package_name][credential_name] = {
                f["field_type"]: f["value"] for f in fields
            }

    async def delete(self, credential_name, *, package_name):
        package_name = package_name.replace(".", "-")
        del self.data[package_name][credential_name]


class Fields:
    def __init__(self, vault):
        self.vault = vault

    async def create(self, credential_name, *, package_name, field_type, value):
        package_name = package_name.replace(".", "-")
        fields = self.vault.data[package_name][credential_name]
        assert field_type not in fields
        fields[field_type] = value

    async def update(self, field_type, *, package_name, credential_name, value):
        package_name = package_name.replace(".", "-")
        fields = self.vault.data[package_name][credential_name]
        assert field_type in fields
        fields[field_type] = value

    async def delete(self, field_type, *, package_name, credential_name):
        package_name = package_name.replace(".", "-")
        del self.vault.data[package_name][credential_name][field_type]


def test_writes_create_update_delete_and_invalidate_cache(caplog):
    async def run():
        vault = CloudCredentials(api_key="dummy", base_url="https://example.invalid")
        fake = FakeVault()
        vault._client = NS(credentials=fake)
        await vault.bind(MemoryDevice())
        assert await vault.list() == []
        await vault.save("app.test", "work", {"password": "hidden-value"})
        assert (await vault.list())[0] == {
            "package": "app.test",
            "name": "work",
            "fields": ["password"],
        }
        await vault.save(
            "app.test", "work", {"password": "changed-value", "email": "a@example.com"}
        )
        await vault.update("app.test", "work", "password", "last-value")
        item = (await vault.list())[0]
        assert set(item["fields"]) == {"password", "email"}
        assert fake.data["app-test"]["work"] == {
            "password": "last-value",
            "email": "a@example.com",
        }
        assert "last-value" not in repr(item)
        await vault.delete("app.test", "work", "email")
        assert (await vault.list())[0]["fields"] == ["password"]
        await vault.delete("app.test", "work")
        assert await vault.list() == []
        assert "hidden-value" not in caplog.text and "last-value" not in caplog.text

    asyncio.run(run())


def test_save_reuses_initialized_empty_package():
    async def run():
        vault = CloudCredentials(api_key="dummy", base_url="https://example.invalid")
        fake = FakeVault()
        fake.data["ai-mobilerun-smoke-test"] = {}
        vault._client = NS(credentials=fake)
        await vault.bind(MemoryDevice())
        await vault.save("ai.mobilerun.smoke.test", "work", {"password": "dummy-value"})
        assert [
            item["name"] for item in await vault.list("ai.mobilerun.smoke.test")
        ] == ["work"]
        await vault.delete("ai.mobilerun.smoke.test", "work")
        assert await vault.list("ai.mobilerun.smoke.test") == []

    asyncio.run(run())


def test_invalid_field_rejected_before_any_write():
    async def run():
        vault = CloudCredentials(api_key="dummy", base_url="https://example.invalid")
        fake = FakeVault()
        vault._client = NS(credentials=fake)
        with pytest.raises(ValueError, match="field type"):
            await vault.save("app.test", "work", {"invalid": "secret"})
        with pytest.raises(ValueError, match="field type"):
            await vault.update("app.test", "work", "invalid", "secret")
        with pytest.raises(ValueError, match="field type"):
            await vault.delete("app.test", "work", "invalid")
        assert fake.data == {}

    asyncio.run(run())


def test_package_read_uses_package_endpoint_after_save():
    async def run():
        vault = CloudCredentials(api_key="dummy", base_url="https://example.invalid")
        fake = FakeVault()
        vault._client = NS(credentials=fake)
        await vault.bind(MemoryDevice())
        await vault.list()
        await vault.save("app.test", "work", {"password": "dummy-value"})
        original_list = fake.list

        async def inconsistent_list(package_name=None, page=None):
            if page is not None:
                return NS(items=[], pagination=NS(has_next=False))
            return await original_list(package_name=package_name)

        fake.list = inconsistent_list

        async def hidden_packages():
            return NS(data=[])

        fake.list_all = hidden_packages
        assert [item["name"] for item in await vault.list("app.test")] == ["work"]

    asyncio.run(run())


@pytest.mark.parametrize(
    ("value", "display"),
    [
        ("4111111111111111", "4111 1111 1111 1111"),
        ("12/27", "12 / 27"),
        ("1227", "12 / 27"),
        ("4826", "4 - 8.2 / 6"),
        ("4111111111111111", "4111  1111 1111 1111"),
        ("1227", "12  /  27"),
    ],
)
def test_formatted_digits_redacted_in_log(caplog, value, display):
    with secret_scope(value):
        logging.getLogger("variant-test").warning("value: %s", display)
    assert caplog.records[-1].getMessage() == "value: ***"


@pytest.mark.parametrize("value", ["id:234567", "api234567"])
def test_alphanumeric_secrets_are_literal_only(value):
    with secret_scope(value):
        assert _redact_secrets(f"{value}; order 234-567") == "***; order 234-567"


@pytest.mark.parametrize("reverse", [False, True])
def test_overlapping_secret_variants_leave_no_suffix(reverse):
    values = ("4111", "4111111111111111", "label", "label-private")
    with secret_scope(*(values[::-1] if reverse else values)):
        assert _redact_secrets("4111 1111 1111 1111 label-private") == "*** ***"


def test_short_numbers_and_longer_unrelated_digit_runs():
    with secret_scope("1/2", "4826"):
        message = "Order 12; transaction 948267; other 94-8267"
        assert _redact_secrets(message) == message
        assert _redact_secrets("literal 1/2") == "literal ***"


def test_lone_surrogate_retains_literal_and_escaped_protection():
    value = "test\ud800"
    with secret_scope(value):
        assert _redact_secrets(value + " " + repr(value)[1:-1]) == "*** ***"


def test_no_active_scope_returns_message_unchanged():
    message = "Order 12; 4111 1111 1111 1111"
    assert _redact_secrets(message) is message


@pytest.mark.parametrize(
    ("value", "message", "expected"),
    [
        ("12/27", "dates 2026/12/27 and 12/27/99", "dates 2026/12/27 and 12/27/99"),
        (
            "1227",
            "dates 2026 / 12 / 27 and 12.27-99",
            "dates 2026 / 12 / 27 and 12.27-99",
        ),
        (
            "123456",
            "order 99-123-456 and 123.456.77",
            "order 99-123-456 and 123.456.77",
        ),
        ("123456", "code 123456, ID 123-456.", "code ***, ID ***."),
        ("12/27", "expires 12 / 27.", "expires ***."),
        ("4826", "PIN 4826. 2 attempts remain", "PIN ***. 2 attempts remain"),
        ("12/27", "12.27.99 and 12/27.", "12.27.99 and ***."),
        ("4826", "items 2  4826  99", "items 2  ***  99"),
        (
            "4111111111111111",
            "order 9 card 4111 1111 1111 1111 2 items",
            "order 9 card *** 2 items",
        ),
    ],
)
def test_numeric_candidate_boundaries(value, message, expected):
    with secret_scope(value):
        assert _redact_secrets(message) == expected


def test_secret_patterns_do_not_enter_process_regex_cache():
    marker = "regexcachecanary723819"
    with secret_scope(marker):
        assert _redact_secrets(marker) == "***"
    assert not any(marker in str(key) for key in re._cache)


def test_large_active_scope_scan_budget():
    text = ("trace message with id 19. " * 2600)[:65536]
    with secret_scope(
        *(str(200000 + i) if i % 2 else f"label-{200000+i}" for i in range(128))
    ):
        start = time.process_time()
        for _ in range(4):
            assert _redact_secrets(text) is text
        assert time.process_time() - start < 0.5
