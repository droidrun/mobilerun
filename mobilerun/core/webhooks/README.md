# Cloud webhook endpoints

Use `CloudWebhooks` to manage the current organization's webhook destinations.
All endpoint methods are async and use the configured cloud credentials.
The service signs HTTP deliveries and returns the signing secret only on initial
registration. Replaying a create with the same idempotency key does not return it again.

```python
from mobilerun.core.webhooks import DEFAULT_WEBHOOK_EVENT, CloudWebhooks

hooks = CloudWebhooks(billing="no_commerce")
endpoint = await hooks.register(
    "https://receiver.example/results",
    event_types=[DEFAULT_WEBHOOK_EVENT],
)
try:
    # Explicitly send one signed test POST to the receiver.
    delivered, http_status = await hooks.ping(endpoint.id)
finally:
    await hooks.delete(endpoint.id)
```

`register` requires an explicit filter. `DEFAULT_WEBHOOK_EVENT` is the flow
results event (`workflow.step.emit`). Passing `event_types=[]` subscribes the
endpoint to **every** deliverable platform event. An optional `idempotency_key`
lets you retry the same registration intent. Store the returned secret securely
when keeping the endpoint; never log it. Endpoint management does not purchase
resources. Registration does not run a flow or send a test delivery; call `ping`
explicitly to send one.

`list()` reads every page, including integration destinations (which have no URL).
`get(endpoint_id)`, `ping(endpoint_id)`, and `delete(endpoint_id)` take an endpoint UUID.
`find_by_url(url)` looks for a non-deleted HTTP endpoint by exact normalized HTTPS
URL without sending it as a search query. Finding an endpoint does not grant
ownership: inspect its `state`, `health`, and `event_types` before deciding to
reuse it. `register` does not adopt
an existing endpoint on conflict. URLs and signing secrets are `SecretStr` values;
reading them in plaintext requires `.get_secret_value()`.

`validate_webhook_url(url)` performs the same validation and canonicalization as registration, without a network request. Registration distinguishes a duplicate URL, an in-progress idempotent request, and a deleted registration on replay through the typed cloud error's `code`.

| File | Purpose |
| --- | --- |
| `__init__.py` | Public API, including `DEFAULT_WEBHOOK_EVENT` |
| `cloud.py` | Endpoint operations, validation, and URL lookup |

Set `MOBILERUN_WEBHOOK_PAGE_SIZE` and `MOBILERUN_WEBHOOK_PING_TIMEOUT` via
[shared settings](../../config/README.md#settings-overview).

`ping` uses the SDK’s `webhooks.test_delivery(id)` (`POST /webhooks/{id}/test`). The service signs and attempts delivery before returning the outcome and HTTP status, so there is no need to poll delivery history. Receiver error text is not returned. The SDK also exposes `deliveries.list_for_webhook` and `deliveries.retrieve_attempts` for history. See the [backend implementation](https://github.com/droidrun/webhooks-api/blob/94783f57f4c7c4e7c014ae3fcaf91c0b39396e09/src/modules/endpoints/service/endpoints.service.ts#L495-L530).

See the [cloud SDK adapter](../../clients/cloud/webhooks.py),
[endpoint model](../../models/webhooks.py), [webhook CLI](../../cli/webhook/README.md),
and [sandbox result delivery](../../sandbox/README.md#send-a-function-result-to-a-webhook).
