# Webhook CLI

Manage the current organization's webhook endpoints with configured cloud credentials. Registration creates a cloud resource but does not purchase one. To keep the URL out of shell history, omit it and enter it at the visible interactive prompt:

```sh
mobilerun webhook register
mobilerun webhook list --json
mobilerun webhook get ID
mobilerun webhook ping ID
mobilerun webhook delete ID --yes
```

Enter the URL. Registration defaults to flow results (`workflow.step.emit`). Use the returned ID with `@cloud("my-function", webhook="<id>")` or as `endpointId` in a flow's `webhooks.send` step. To override the default, pass `--event-type EVENT` once per event; the backend validates the filter. Deploying and running a [sandbox flow](../../sandbox/README.md#send-a-function-result-to-a-webhook) can incur charges.

The interactive prompt echoes the URL as you type. With no ID, `get`, `ping` and `delete` let you pick an endpoint in a terminal. In noninteractive use, pass the endpoint ID and use `--yes` to confirm deletion. Delete only an endpoint you own or are authorized to remove. `ping` sends a signed test POST, reports the receiver's HTTP status if available, and exits nonzero if delivery fails.

A matching URL returns the existing ID without changing the endpoint or revealing a signing secret. If the endpoint is inactive, blocked, or excludes any requested event, registration exits nonzero and explains why. On creation, store the signing secret immediately. `register URL --json` includes it in `secret` on stdout, so keep stdout out of logs. If a create is replayed without its secret, the SDK's `client.webhooks.rotate_secret(id)` replaces it immediately; update the receiver to verify with the new secret. List, get and the endpoint picker show the URL path, but [mask query values](../../utils/README.md) (`?token=•••`). Paths may themselves be sensitive; share output carefully.

| File | Purpose |
| --- | --- |
| `group.py` | Register, list, inspect, ping, and delete endpoints |
| `__init__.py` | CLI package marker |

See the [core API](../../core/webhooks/README.md) and [CLI command tree](../README.md).
