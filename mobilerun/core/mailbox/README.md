# Mailboxes

## Quick example

A cloud mailbox receives email and extracts codes.

Creating a mailbox may incur charges. This waits for an email sent to its address.

```python
from mobilerun.core.mailbox import CloudMailbox

async with CloudMailbox(billing="paid") as mailbox:
    print(mailbox.address)
    code = await mailbox.wait_for_code(timeout=60)
    print(code)
```

See [phone numbers](../phone_number/README.md) and [extensions](../extensions/README.md).

Creation waits up to 30 seconds for an asynchronously prepared checkout URL. Non-interactive
creation raises `PaymentRequired` with the URL and optional expiry, and leaves the pending
mailbox intact. Resume using `await error.wait_until_paid(timeout=600)` to receive
the ready mailbox; resumed waits never cancel on timeout. Alternatively pass
`on_checkout=callback` when creating the mailbox. The callback receives a typed
`Checkout(url, expires_at, resource, resource_id)` once, and may be async.
Alternatively, `checkout="print"` prints the URL, and `checkout="browser"` also opens
it when a graphical display is available. Both wait for payment and cannot be combined
with `on_checkout`. The default, `checkout="error"`, does not print or open a browser.
`payment_timeout` defaults to 600 seconds. The CLI and demo use browser mode;
if payment times out or is interrupted they cancel the pending resource. A preparation
timeout names the resource and asks you to check the dashboard.

`Mailbox` defines address and code waiting. Await `CloudMailbox(billing=...)` to
create an owned mailbox, or pass an id to retrieve one. `keep=True` retains it.

There is no server mailbox binding API. `await mailbox.bind(device)` and
`await mailbox.unbind()` attach and detach a local extension.
This works on cloud and local typing devices, including through wrappers.
`phone.mailbox` is read-only. Code polling shares the SMS timeout helper.

### Included quota and billing plan

`await CloudMailbox.included_available()` returns remaining included mailboxes;
it is a snapshot, not a reservation. Create with `plan="included"` to avoid
renting, default `plan="auto"` to use included quota first (renting only when
billing is paid), or `plan="rent"` to always rent with `billing="paid"`.
Unlike cloud phone `plan="auto"`, resource `auto` also accepts included billing.
Import `ResourcePlan` from `mobilerun.models` for enum values.
