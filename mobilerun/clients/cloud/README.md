# cloud/

**Purpose:** The only SDK boundary for cloud services, including billing policy.

**What belongs here / what doesn't:** Put cloud account access and OAuth flows here, not CLI presentation.

**Contents:**

- `__init__.py` — Package marker.
- `api_keys.py` — API key validation.
- `device_flow.py` — OAuth device authorization flow.
- `client.py` — Shared AsyncMobilerun factory.
- `billing.py` — Shared fail-closed commerce guard.
- `checkout.py`, `payments.py` — Checkout presentation and payment polling.
- `errors.py` — Sanitized cloud API errors.
- `devices.py`, `numbers.py`, `mailboxes.py`, `proxies.py` — Resource operations.
- `secrets.py` — Credential-vault SDK operations; core owns caching and package policy.
- `webhooks.py` — Endpoint registration, lookup, listing, deletion, and signed test delivery for [CloudWebhooks](../../core/webhooks/README.md).
- `tasks.py` — Task calls, streaming, status and artifact SDK access.
- `apps.py` — APK upload, reuse and retrieval.

**Examples:**

```python
from mobilerun.models import Billing

billing = Billing.NO_COMMERCE
```

**Depends on / used by:** Uses shared configuration and HTTP dependencies; used by cloud devices and authentication commands.

## App library

`upload_app(path, billing=Billing.NO_COMMERCE)` inspects an APK with Android build-tools `aapt`, hashes it,
reuses an available matching package and version, or uploads it to the cloud
library. Install Android SDK build-tools and set `ANDROID_HOME`/`ANDROID_SDK_ROOT`
or put `aapt` on `PATH`. The helper accepts optional `api_key` and `base_url`;
otherwise it uses the usual cloud credential resolution. Existing matching
versions are reused without creating app storage.

```python
from mobilerun.clients.cloud.apps import upload_app
from mobilerun.models import Billing

app = await upload_app("counter.apk", billing=Billing.NO_COMMERCE)
```

## Billing

Every public client function requires keyword-only `billing: Billing`, without a
default. Enum values and their strings work; `None` does not. Each boundary declares
its minimum with `@requires(Billing.X)`. Missing billing raises `TypeError`, an
invalid mode raises `BillingError`, and a lower mode raises `BillingRequired`
with `needed`, `given`, and `operation`, all before transport. The order is
`NO_COMMERCE < INCLUDED < PAID`. Import enums from `mobilerun.models` (clients)
or `mobilerun.core.devices` (application code).

Core resources snapshot `MOBILERUN_BILLING` (default `paid`) at construction unless
`billing=` is explicit, and pass that mode on every client call. Cloud CLI commands accept
`--billing no_commerce|included|paid`, defaulting to that setting.

For a single operation whose guard runs before side effects, call
`await with_billing_retry(operation, billing, yes=False)` from
`mobilerun.clients.cloud.billing`. The callback receives the mode for each attempt;
the helper returns `(result, effective_billing)`. It prompts on a terminal only if
the guard rejects the requested mode. `yes=True` authorizes one retry with the
mode required by the guard without a prompt. Only a retry with `PAID` can charge.
It does not retry checkout, transport errors,
or an exhausted included quota. Do not wrap a batch or a callback that may do work
before hitting a billing guard.

| Function | Minimum mode |
| --- | --- |
| `client._cloud_client` | `NO_COMMERCE` |
| `api_keys.validate_api_key` | `NO_COMMERCE` |
| `device_flow.request_code`, `poll_token`, `fetch_session` | `NO_COMMERCE` |
| `devices.create` | `INCLUDED`; `PAID` for `AUTO`/`MINUTE` or a country Connect proxy |
| `devices.resume` | `NO_COMMERCE` for subscription-billed phones; `PAID` otherwise |
| `devices.list_devices`, `retrieve`, `wait_ready`, `terminate`, `stop`, `reboot`, `reset`, `number`, `bind_number`, `unbind_number` | `NO_COMMERCE` |
| `numbers.create`, `mailboxes.create` | `INCLUDED` |
| `numbers.list_numbers`, `retrieve`, `delete`, `countries`, `capacity`, `messages` | `NO_COMMERCE` |
| `numbers.send` | `PAID` |
| `mailboxes.list_mailboxes`, `retrieve`, `delete`, `capacity`, `messages`, `otp` | `NO_COMMERCE` |
| `proxies.connect_device` | `NO_COMMERCE` for an existing Connect ID or SOCKS5; `PAID` for country Connect |
| `proxies.buy_proxy` | `PAID` |
| `proxies.disconnect_device`, `device_status`, `list_proxies`, `get_proxy` | `NO_COMMERCE` |
| `secrets.list_secrets`, `list_package`, `create_package`, `save`, `create_field`, `update`, `delete` | `NO_COMMERCE` |
| `webhooks.register_webhook`, `list_webhooks`, `get_webhook`, `ping_webhook`, `delete_webhook` | `NO_COMMERCE` |
| `apps.get_app`, `upload_app`, `list_installs`, `install`, `uninstall` | `NO_COMMERCE` |
| `files.upload_file`, `list_files`, `download_file`, `delete_file`, `push_device_file`, `list_device_files`, `download_device_file`, `delete_device_file` | `NO_COMMERCE` |
| `tasks.run` | `INCLUDED` (consumes task credits) |
| `tasks.progress`, `status`, `stop`, `trajectory`, `screenshots` | `NO_COMMERCE` |

| Mode | Phones | Numbers / mailboxes | Connect proxy |
| --- | --- | --- | --- |
| `no_commerce` | Blocked | Blocked | Blocked |
| `included` | `subscription`, no fallback | `included_only` | Blocked: no SDK-enforced included-only mode |
| `paid` (default) | Any plan; default `auto` | `included` (paid fallback) | Allowed |

Existing IDs never trigger a purchase. A Connect proxy ID can be attached without
buying a proxy; selecting a Connect country may purchase one and is guarded.
`proxies.list_proxies(included_only=True, billing=Billing.NO_COMMERCE)` filters
each page to proxies billed as included. Pagination still describes the unfiltered
inventory, so a filtered page can be empty even when more pages are available.
The old allow-paid setting and resource create classmethods are removed.

App uploads and installs require the explicit mode, but permit `NO_COMMERCE`.

```python
app = await device.install_apk("counter.apk")
await device.start_app(app.package)
await device.uninstall_app(app.package)
```

See [clients](../README.md) and [cloud devices](../../core/devices/README.md).
