# Phone numbers

A cloud number receives SMS and can bind to a compatible cloud phone.

## Quick example

Creating a number may incur charges. This waits for an SMS sent to the new number.

```python
from mobilerun.core.phone_number import CloudPhoneNumber

async with CloudPhoneNumber(country="US", billing="paid") as number:
    print(number.phone_number)
    code = await number.wait_for_code(timeout=60)
    print(code)
```

See [devices](../devices/README.md) and [mailboxes](../mailbox/README.md).

## Number readiness

`await number.wait_ready()` polls `numbers.retrieve(id).data.state` until
`active`. The SDK states are `awaiting_payment`, `provisioning`, `active`,
`cancel_scheduled`, `expired` and `failed`. Only `provisioning` is polled;
`awaiting_payment` raises `PaymentRequired(BillingError)` immediately, including
the resource ID and checkout URL when provided. A non-interactive pending purchase is
not cancelled. Creation polls for an asynchronously prepared URL. Pass
`on_checkout=callback` to receive typed `Checkout` details once and wait for
activation (async callbacks are awaited). Callback mode cancels on timeout or
interruption; `payment_timeout` defaults to 600 seconds. Without a callback,
`await error.wait_until_paid(timeout=600)` returns the ready number without
cancelling on timeout. Alternatively use `checkout="print"` to print the URL or `checkout="browser"`
to print it and open a browser where available; both wait and cancel on timeout
or interruption. Neither can be combined with `on_checkout`. The default
`checkout="error"` does not print or open a browser. The CLI uses browser mode.
Terminal/cancellation states fail immediately. A timeout reports the last state.
The number's cached `phone_number` is refreshed on activation. Progress logs at
DEBUG include the resource ID, state and elapsed seconds, never checkout URLs.

`MOBILERUN_CLOUD_NUMBER_READY_TIMEOUT_SECONDS` sets the positive timeout (default
600 seconds); `wait_ready(timeout=...)` overrides it per call. This is a
conservative client budget, **not a documented provisioning SLA**. The
[purchase API](https://docs.mobilerun.ai/api-reference/numbers/purchase-a-phone-number)
requires polling but does not specify a completion time. The
[retrieve API](https://docs.mobilerun.ai/api-reference/numbers/get-phone-number-by-id)
defines the states. `awaiting_payment` requires a checkout decision; it is not
slow provisioning. Waiting never authorizes or completes payment. Device provisioning and waiting for an
SMS are separate operations with different timeouts.

`PhoneNumber` defines the interface. `CloudPhoneNumber` creates an owned number
when awaited, or retrieves an existing number by id. Creation accepts `billing`,
`country` and `keep`.

Use `await number.bind(phone)` and `await number.unbind()`. These are the only
binding operations. `phone.number` is read-only. Local devices reject binding
before making requests. Cloud phones must support telephony.
Await an existing `CloudDevice(id)` to fetch/cache its
server-side number binding. Use `wait_for_code()` to retrieve a code explicitly.

Every `Device` declares the read-only `number` property. A local device or a
cloud phone without a bound number raises `DeviceOperationNotSupported` with
support/attachment guidance. After binding, use the typed resource directly:

```python
code = await device.number.wait_for_code(sender="Example", timeout=60)
messages = await device.number.messages()
await device.number.send("+15555550123", "Test message")
```

Wrappers delegate `number`; `DevicePool.number` returns resources in device order.

Owned numbers bound to an owned phone are unbound and released when the phone
closes, unless the phone or number has `keep=True`.

### Included quota and billing plan

`await CloudPhoneNumber.included_available("US")` returns the remaining included
numbers for a country. This is a snapshot, not a reservation.
`await CloudPhoneNumber(country="US", billing="paid", plan="included")` uses
only included quota. The default `plan="auto"` uses included quota first and
rents only with `billing="paid"`; `plan="rent"` always rents and requires paid
billing. Unlike a cloud phone's explicit `plan="auto"`, resource `auto` also
works with `billing="included"`. Import `ResourcePlan` from `mobilerun.models`
to use enum values instead of strings.
