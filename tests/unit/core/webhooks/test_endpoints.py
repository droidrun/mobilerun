"""Exercise the core API against the generated SDK with an offline HTTP transport."""

import json
from uuid import UUID

import httpx
import pytest
from mobilerun_sdk import AsyncMobilerun

from mobilerun.errors.cloud import CloudResourceNotFound, CloudValidationError

ID = str(UUID(int=7))
URL = "https://receiver.example/path?token=test-sentinel"
SIGNATURE = "test-signing-value"


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("https://host.example/" + "x" * 2048 + "sentinel", "exceeds"),
        ("http://sentinel.example/", "must start with https://"),
        ("https://sentinel.example/a b", "whitespace"),
        ("https://sentinel.example/a\x7fb", "control"),
        ("https://sentinel@host.example", "username or password"),
        ("https://host.example/#sentinel", "fragment"),
        ("https://sentinel.example:bad/", "hostname and port"),
        ("https:///sentinel", "hostname and port"),
    ],
)
def test_url_validation_reasons_never_include_input(value, reason):
    from mobilerun.core.webhooks import validate_webhook_url

    with pytest.raises(CloudValidationError) as error:
        validate_webhook_url(value)
    assert reason in str(error.value)
    assert "sentinel" not in str(error.value) and error.value.__context__ is None


def test_url_rejects_percent_encoding_that_exceeds_length_limit():
    from mobilerun.core.webhooks import validate_webhook_url

    # Each é becomes six characters; the canonical URL would be 2049 characters.
    value = "https://receiver.example/" + "é" * 337 + "ab"
    with pytest.raises(CloudValidationError) as error:
        validate_webhook_url(value)
    assert str(error.value) == "Webhook URL exceeds 2048 characters."


def test_url_accepts_canonical_length_at_limit_after_trimming_slash():
    from mobilerun.core.webhooks import validate_webhook_url

    value = "https://receiver.example/" + "é" * 337 + "a/"
    canonical = validate_webhook_url(value)
    assert canonical == "https://receiver.example/" + "%C3%A9" * 337 + "a"
    assert len(canonical) == 2048
    assert validate_webhook_url(canonical) == canonical


@pytest.fixture
def service(monkeypatch):
    from mobilerun.clients.cloud import webhooks
    from mobilerun.core.webhooks import CloudWebhooks

    requests = []
    clients = []
    row = dict(
        id=ID,
        url=URL,
        kind="http",
        state="ACTIVE",
        health="healthy",
        eventTypes=["workflow.step.emit"],
        signingEnabled=True,
        createdAt="now",
        updatedAt="now",
    )

    def handler(req):
        requests.append(req)
        if req.url.path.endswith("/test"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "success": False,
                        "statusCode": 503,
                        "error": URL + SIGNATURE,
                    }
                },
            )
        if req.method == "POST":
            return httpx.Response(201, json={"data": {**row, "secret": SIGNATURE}})
        if req.method == "DELETE":
            return httpx.Response(204)
        if req.url.path.endswith("/webhooks"):
            page = int(req.url.params["page"])
            return httpx.Response(
                200,
                json={
                    "items": (
                        [row] if page == 2 else [{**row, "url": URL + "-different"}]
                    ),
                    "pagination": {"hasNext": page == 1},
                    "counts": {},
                },
            )
        if not req.url.path.endswith(ID):
            return httpx.Response(404, json={"message": URL + SIGNATURE})
        return httpx.Response(200, json={"data": row})

    def factory(*, billing):
        client = AsyncMobilerun(
            api_key="offline",
            base_url="https://api.example/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(webhooks, "cloud_client", factory)
    return CloudWebhooks(billing="no_commerce"), requests, clients


@pytest.mark.anyio
async def test_register_lookup_delete(service, caplog):
    import logging

    from mobilerun.core.webhooks import DEFAULT_WEBHOOK_EVENT

    caplog.set_level(logging.DEBUG, logger="mobilerun_sdk")
    manager, requests, clients = service
    created = await manager.register(
        URL, event_types=[DEFAULT_WEBHOOK_EVENT], idempotency_key="stable"
    )
    assert created.id == ID and created.secret.get_secret_value() == SIGNATURE
    assert created.url.get_secret_value() == URL
    assert URL not in repr(created) and SIGNATURE not in repr(created)
    assert URL not in caplog.text and SIGNATURE not in caplog.text
    assert requests[0].headers["Idempotency-Key"] == "stable"
    assert json.loads(requests[0].content) == {
        "url": URL,
        "eventTypes": ["workflow.step.emit"],
    }
    assert (await manager.get(ID)).secret is None
    assert len(await manager.list()) == 2
    found = await manager.find_by_url(
        URL.replace("receiver.example", "RECEIVER.example:443").replace(
            "path?", "path/?"
        )
    )
    assert found and found.id == ID
    assert await manager.find_by_url(URL + "-missing") is None
    assert all("token=" not in str(req.url) for req in requests)
    await manager.delete(ID)
    assert requests[-1].method == "DELETE" and requests[-1].url.path.endswith(ID)
    assert all(client.is_closed() for client in clients)


@pytest.mark.anyio
async def test_list_uses_configured_page_size_for_every_page(service, monkeypatch):
    manager, requests, _ = service
    monkeypatch.setenv("MOBILERUN_WEBHOOK_PAGE_SIZE", "23")

    assert len(await manager.list()) == 2
    assert [request.url.params["pageSize"] for request in requests] == ["23", "23"]


@pytest.mark.anyio
async def test_invalid_inputs_and_error_privacy(service):
    manager, requests, _ = service
    for bad in [
        URL.replace("https:", "http:"),
        URL + "#fragment",
        "https://a:b@example.com",
        "not-a-url",
    ]:
        with pytest.raises(CloudValidationError) as error:
            await manager.register(bad, event_types=[])
        assert bad not in str(error.value)
    with pytest.raises(CloudValidationError):
        await manager.delete(URL)
    assert not requests
    with pytest.raises(CloudResourceNotFound) as error:
        await manager.get(str(UUID(int=8)))
    assert error.value.__context__ is None
    assert URL not in repr(error.value) and SIGNATURE not in repr(error.value)


@pytest.mark.anyio
async def test_ping_uses_signed_backend_test_and_drops_receiver_error(service):
    manager, requests, clients = service
    with pytest.raises(CloudValidationError):
        await manager.ping(URL)
    assert not requests
    assert await manager.ping(ID) == (False, 503)
    assert [(r.method, r.url.path) for r in requests] == [
        ("POST", f"/v1/webhooks/{ID}/test")
    ]
    assert all(c.is_closed() for c in clients)
