# Proxy extensions

Proxies are standalone extensions, not device settings methods:

## Quick example

With an existing connected Android device `android`:

```python
from mobilerun.core.proxy import LocalProxy

proxy = LocalProxy("localhost", 8080)
await proxy.bind(android)
assert android.proxy is proxy
status = await proxy.status()
await proxy.unbind()
assert proxy.device is None
```

The default extension name is `proxy`; `phone.proxy` returns the proxy object
itself. Keep the original handle for static typing; there is no typed lookup API.
The device's registry is read-only and shared through wrappers.

| Class | Compatibility | Exclusive kind |
| --- | --- | --- |
| `CloudRotatingProxy(country, smart_ip=True)` | Cloud device | `proxy` |
| `CloudFixedProxy(id)` | Cloud device | `proxy` |
| `CloudSocks5Proxy(host, port, user, password)` | Cloud device | `proxy` |
| `ExistingCloudProxy()` | Observes an active, unidentified cloud proxy | `proxy` |
| `LocalProxy(host, port)` | Local Android with `http_proxy` support | `proxy` |

All cloud variants inherit the shared `CloudProxy` base. Compatibility is checked
by the extension using declared device facts, never concrete device class tests.
Credentials are excluded from repr, context and action results. `status()` is a
public typed action returning `ProxyStatus(connected, name, protocol)`.

## Conflicts and switching

An unmanaged active cloud proxy raises `mobilerun.errors.UnmanagedProxyConflict`,
a subclass of `ExtensionConflict`, unless the bind explicitly uses `force=True`.
It is also re-exported from `mobilerun.core.proxy`.

Binding a second proxy raises `ExtensionConflict`, naming the bound extension,
**before any SDK mutation**. There is no automatic replacement. For local devices,
unbind the old proxy before binding another, or explicitly force replacement:

```python
await replacement.bind(device, force=True)
```

Force only replaces same-kind exclusivity conflicts. Compatibility failures,
unrelated name collisions and SDK errors still raise. It unbinds the old extension
without closing or releasing it, even if device cleanup previously owned it.

Cloud phones require a proxy. New cloud-device creation uses the supplied typed
`CloudProxy`, or configured rotating defaults, and publishes that exact instance
as a bound extension after creation. No second provisioning request is made.
Passing `None` is invalid. Unbinding the only cloud proxy is rejected: switch with
`new_proxy.bind(phone, force=True)` instead. Closing a borrowed cloud device removes
the local relationship but leaves its required server proxy configured.

An unmanaged active cloud proxy also requires explicit `force=True` takeover.
Cloud-to-cloud replacement uses one API switching call where supported, not a
disconnect followed by connect. On failure, the coordinator attempts to restore
the old proxy. A failed initial bind or any rollback triggers read-only status
reconciliation **before the original error or cancellation is raised**. The registry
then reflects the remotely active proxy, not the requested configuration. Failed
handles are detached; observing a proxy does not grant ownership of its capacity.
Reconciliation survives repeated caller cancellation and has a 10-second bound.
If status cannot be read, a relationship remains marked uncertain, ordinary actions
fail clearly, and the original error includes a diagnostic note. Reconnect or bind
another proxy with `force=True` to recover. An API error is not a guarantee of
network rollback: some backends disconnect the old proxy before the new one fails.

After a successful restore, a positive identity match preserves the original
handle and ownership metadata. Policy for an empty remote name: a connected proxy
with no identity also preserves the successfully restored handle. A different,
nonempty but unresolvable reference is not a match: it becomes `ExistingCloudProxy`.

## Existing cloud phones: read-only adoption

`await CloudDevice(existing_id)` reads remote proxy status and publishes an
extension without provisioning, switching, disconnecting, or purchasing anything:

- A `mrc:<id>` name becomes `CloudFixedProxy(id)`. Other nonempty references are
  resolved with a read-only Connect lookup when possible.
- Connected but unidentified (including an empty name) becomes
  `ExistingCloudProxy`, a typed **existing proxy (unknown)** observation. Its
  `status()` action works, but it contains no reusable credentials and cannot be
  used to create a cloud phone or reapply an unidentified configuration.
- Disconnected means **no proxy extension**. Connect does not silently repair the
  phone. Accessing `phone.proxy` raises an `AttributeError` explaining that there is
  no active proxy and suggesting `await proxy.bind(phone)`.

Both known and unknown observations are exclusive `"proxy"` extensions. They reject
standalone unbind on a cloud phone; replace them with
`await replacement.bind(phone, force=True)`. A failed status read raises instead of
claiming that the phone is disconnected. Successful repeated connects reuse the
observation; reconnecting an uncertain binding reads status again.

The API resolves an existing Connect proxy ID to credentials; it does not establish
a demonstrated exclusive cross-device lease. Separate handles for the same ID may
therefore work on different devices. Backend errors are surfaced, not invented.

## CLI and creation

```console
mobilerun proxy bind PROXY_ID DEVICE_ID --force
mobilerun proxy buy --country US --type dedicated_residential --billing paid
```

`bind` references an existing ID and does not purchase proxy capacity. Country-based
rotating proxies may provision capacity and require the existing billing approval.
Only rotating proxies expose smart-IP settings. Configured defaults use
`MOBILERUN_CLOUD_PROXY_COUNTRY` and `MOBILERUN_CLOUD_PROXY_SMART_IP`.

`buy_proxy` checkout, timeout and cancellation behavior is unchanged. Without a
checkout callback, `PaymentRequired` can be resumed without cancelling on timeout.
Callbacks use typed checkout details. By default (`checkout="error"`) the library
never prints or opens a browser; `checkout="print"` / `checkout="browser"` opt in.

Breaking change: `Device.set_proxy`, `clear_proxy`, and `proxy_status` are removed,
including wrapper and pool variants. Pools bind separate extension instances via
`bind_each`; `pool.actions.call("proxy.status", {})` broadcasts explicitly.

See [devices](../devices/README.md) and [extensions](../extensions/README.md).
