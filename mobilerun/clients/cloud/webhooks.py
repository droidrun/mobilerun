"""Guarded webhook API calls. Endpoint management does not purchase resources."""

import asyncio
import re
from collections.abc import Awaitable, Sequence
from typing import TypeVar

import httpx
from mobilerun_sdk import APIError, APIStatusError
from mobilerun_sdk.types.webhook_create_response import Data as Created
from mobilerun_sdk.types.webhook_list_response import Item
from mobilerun_sdk.types.webhook_retrieve_response import Data as Retrieved
from pydantic import SecretStr

from mobilerun.clients.cloud.billing import requires
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.clients.cloud.errors import cloud_error
from mobilerun.errors.cloud import (
    CloudConflict,
    CloudResourceNotFound,
    CloudTimeout,
    CloudValidationError,
)
from mobilerun.models.billing import Billing
from mobilerun.models.webhooks import WebhookEndpoint
from mobilerun.utils.redaction import secret_scope

T = TypeVar("T")


def _registration_error(body: object) -> CloudValidationError:
    # Match only known service messages (endpoints.service.ts and lib/url.ts).
    # Never render the response text, even if it starts with a recognized reason.
    message = body.get("message") if isinstance(body, dict) else None
    reasons = {
        "eventTypes contains an unknown or non-webhook-deliverable type": "The filter contains an unknown or non-webhook-deliverable event type.",
        "url is not a valid absolute URL": "The destination is not a valid absolute URL.",
        "url scheme must be http or https": "The destination needs an HTTPS URL.",
        "url must not contain userinfo (user:pass@…)": "Remove the username or password from the URL.",
        "url must not contain a fragment (#…)": "Remove the URL fragment.",
    }
    reason = "The service rejected this registration; unrecognized details were withheld for privacy."
    if isinstance(message, str):
        if re.fullmatch(r"endpoint limit reached \([0-9]+ per org\)", message):
            reason = "The organization has reached its webhook endpoint limit."
        else:
            reason = reasons.get(message, reason)
    return CloudValidationError(
        f"Register webhook failed (HTTP 400). {reason}",
        operation="Register webhook",
        status=400,
    )


async def _request(request: Awaitable[T], operation: str) -> T:
    try:
        return await request
    except (APIError, httpx.HTTPError) as exc:
        error = cloud_error(exc, operation)
        if (
            operation == "Register webhook"
            and isinstance(exc, APIStatusError)
            and exc.status_code == 400
        ):
            error = _registration_error(exc.body)
        elif (
            operation == "Register webhook"
            and isinstance(exc, APIStatusError)
            and isinstance(exc.body, dict)
        ):
            code = exc.body.get("code")
            if (
                exc.status_code == 409
                and code == "conflict"
                and exc.body.get("message") == "a webhook for this URL already exists"
            ):
                error = CloudConflict(
                    operation=operation, status=409, code="webhook_duplicate_url"
                )
            elif exc.status_code == 409 and code == "idempotency_request_in_progress":
                error = CloudConflict(
                    operation=operation,
                    status=409,
                    code="webhook_registration_in_progress",
                )
            elif (
                exc.status_code == 404
                and code == "not_found"
                and exc.response.headers.get("Idempotent-Replayed") == "true"
            ):
                error = CloudResourceNotFound(
                    operation=operation, status=404, code="webhook_replay_deleted"
                )
    # Drop the raw response and request (which can contain the URL or secret).
    raise error from None


def _endpoint(
    data: Created | Retrieved | Item, secret: str | None = None
) -> WebhookEndpoint:
    return WebhookEndpoint(
        data.id,
        data.kind,
        SecretStr(data.url) if data.url else None,
        data.state,
        data.health,
        tuple(data.event_types),
        SecretStr(secret) if secret else None,
    )


@requires(Billing.NO_COMMERCE)
async def register_webhook(
    url: str,
    event_types: Sequence[str],
    idempotency_key: str | None,
    *,
    billing: Billing,
) -> WebhookEndpoint:
    """Create an HTTP endpoint; never adopt an existing URL on conflict."""
    with secret_scope(url):
        async with cloud_client(billing=billing) as client:
            data = (
                await _request(
                    client.webhooks.create(
                        url=url,
                        event_types=list(event_types),
                        idempotency_key=idempotency_key,
                    ),
                    "Register webhook",
                )
            ).data
            return _endpoint(data, data.secret)


@requires(Billing.NO_COMMERCE)
async def list_webhooks(page_size: int, *, billing: Billing) -> list[WebhookEndpoint]:
    """Read all pages within one SDK client lifetime."""
    endpoints: list[WebhookEndpoint] = []
    async with cloud_client(billing=billing) as client:
        page = 1
        while True:
            response = await _request(
                client.webhooks.list(page=page, page_size=page_size), "List webhooks"
            )
            endpoints.extend(_endpoint(item) for item in response.items)
            if not response.pagination.has_next:
                return endpoints
            page += 1


@requires(Billing.NO_COMMERCE)
async def get_webhook(endpoint_id: str, *, billing: Billing) -> WebhookEndpoint:
    """Read one endpoint; the service never returns its signing secret here."""
    async with cloud_client(billing=billing) as client:
        return _endpoint(
            (await _request(client.webhooks.retrieve(endpoint_id), "Get webhook")).data
        )


@requires(Billing.NO_COMMERCE)
async def ping_webhook(
    endpoint_id: str, timeout: float, *, billing: Billing
) -> tuple[bool, int | None]:
    """Read the synchronous test result; receiver error strings may contain secrets."""
    async with cloud_client(billing=billing) as client:
        try:
            async with asyncio.timeout(timeout):
                data = (
                    await _request(
                        client.with_options(max_retries=0).webhooks.test_delivery(
                            endpoint_id
                        ),
                        "Ping webhook",
                    )
                ).data
        except TimeoutError:
            raise CloudTimeout(operation="Ping webhook") from None
    return data.success, int(data.status_code) if data.status_code is not None else None


@requires(Billing.NO_COMMERCE)
async def delete_webhook(endpoint_id: str, *, billing: Billing) -> None:
    """Delete the explicitly selected endpoint."""
    async with cloud_client(billing=billing) as client:
        await _request(client.webhooks.delete(endpoint_id), "Delete webhook")
