# Extensions

An extension is a standalone resource with typed async `@action` methods and
public `ContextItem` metadata. Bind explicitly with `await extension.bind(device)`.
The device owns only the shared registry; the extension owns compatibility checks.

Repository test helpers live in `tests/support`: `MemoryDevice` in `devices.py`
and `check_extension_contract` in `extensions.py`. The contract checker covers
binding invariants and safe standalone/device-action probes. These helpers are
internal to the repository and are not shipped in the package.

`check_bind` calls `super()` and returns conflicts with the same `exclusive` key.
Conflicts fail before backend calls. `force=True` unbinds conflicts without closing
or releasing them; compatibility and unrelated name collisions still fail.
Device cleanup unbinds borrowed extensions and closes extensions bound with
`close_on_device_close=True`, before closing its transport.

Use the original typed handle. There is no typed `Device.get` API. Dynamic
`device.name` resolves only registered names; unknown names raise AttributeError.
Discover actions through `device.actions` and public metadata through
`await device.context()`. Action arguments/results are schema-validated. Secret
values must never be returned or included in context.

## Quick example: standalone and device actions

`@action` runs bound or unbound, through direct typed calls or the extension's
`actions.call(...)` catalog. Use it for vault listings, mailbox codes, and number
messages. `@device_action` always requires a binding and injects the bound facade
before the body runs:

```python
from mobilerun.core.devices import Device
from mobilerun.core.extensions import Extension, action, device_action

class Example(Extension):
    @action
    async def available(self) -> bool:
        """Read standalone resource metadata."""
        return True

    @device_action
    async def write(self, device: Device, text: str) -> None:
        """Type through the bound recording/stealth facade."""
        await device.input_text(text)

# await example.write("hello")  # device is injected, never passed by the caller
```

The first parameter after `self` must be `device: Device`. It is hidden from
the typed public signature, runtime signature, and input schema. Unbound calls
raise `ExtensionNotBound("'example.write' needs a bound device; await example.bind(device) first")`
before executing the body. Listings expose `requires_device: bool`, even unbound.
Both decorators support bare or configured forms (`@action()`,
`@device_action(public=False, sensitive_parameters=frozenset({"text"}))`).
Overrides inherit the original decorator unless explicitly redecorated.

For a cloud operation that checks billing before any side effects, use
`@action(billing_retry=_send_with_billing)`, where the handler accepts
`(self, policy: BillingRetry, /, ...)` followed by the ordinary action arguments.
The policy is bound to this handler only, never serialized or recorded.
Callers may pass it to `actions.call(..., billing_retry=policy)` only for entries
with `supports_billing_retry=True`. Pass the callback only the guarded operation
and its starting billing mode; keep validation and the action lock around both
attempts. With no policy, run once at the action's configured mode. A replacement
method must opt in explicitly; it never inherits retry support from the old body.

Bound actions still reject uncertain bindings and draining devices. Unbound
actions retain serialization, validation, safe error wrapping, and nested-call
rules. Local proxy status requires a device; cloud proxy status returns a
disconnected status when unbound and queries remote status when bound.

## Server-side cloud task contract

`Extension.cloud_native: ClassVar[bool] = False` declares whether a server-side
cloud task can use an extension. Third-party extensions default to false.
Cloud proxies opt in because their configuration is already device state on the
server. CloudMailbox remains false: the task API has no mailbox input.

```python
from mobilerun.models import CloudCredentialReference, CloudTaskInputs

async def cloud_task_inputs(self) -> CloudTaskInputs:
    return CloudTaskInputs(
        credential_references=(CloudCredentialReference("com.example", "login"),)
    )
```

The default implementation returns `CloudTaskInputs()`. Both models are frozen;
references contain only `package` and `name` strings, never credential values.
The cloud task operator merges references from all bound `cloud_native` extensions,
deduplicating names per package. Non-native extensions are not asked for inputs.
No concrete extension-class checks or vault-specific Extension hooks are needed.

Action failures remain safely converted to `ExtensionActionError`; there is no
`propagate_exceptions` option. The exception is public `MobilerunError` subclasses
that explicitly declare `value_safe: ClassVar[bool] = True`: actions re-raise them
unchanged so callers can catch typed failures such as `CredentialFieldNotFound`.
Opt in only when every message (including subclass messages) contains no values.
All other failures stay wrapped without backend details.
Wrapped failures expose `ExtensionActionError.cause_type` and include the exception
type plus a value-free category in the message. Timeouts include a finite,
nonnegative numeric `timeout` argument in seconds only when it is not sensitive.
Framework errors without an underlying cause leave `cause_type` empty.

By default, wrapped errors suppress exception chaining. With the `mobilerun` logger
at DEBUG, nonsensitive actions retain the original cause for diagnostics. Sensitive
actions and their nested calls never attach unsafe backend causes, even at DEBUG.
Opted-in value-safe typed errors retain their identity and attributes. Their
implicit contexts and unsafe explicit causes are stripped, including at DEBUG;
only acyclic value-safe explicit causes survive. Cloud errors opt in per instance
only when constructed with keyword diagnostic fields, not legacy free-text messages.
Backend exceptions from bind/unbind hooks propagate
unchanged. Do not return secrets in action results, context, or cloud task inputs.

Cross-extension nested action calls are rejected before acquiring another action
lock. Same-extension nested actions remain supported. Sensitive parameters cannot
have defaults; exported schemas remove defaults, examples, and constants from their
parameter schemas and reachable definitions. Failed or cancelled detach hooks
leave uncertain bindings until read-back reconciles the actual proxy state.
Devices and wrappers explicitly reject copying and pickling.

`close()` marks the extension closing before waiting for in-flight actions, then
releases its resource under the action lock. New and queued actions fail with the
value-safe `ExtensionClosed`; an already-running action can finish same-extension
nested work. Refused detach validation and cancellation before mutation leave the
handle usable. Once detach/cleanup mutation begins, failed or cancelled cleanup
stays closing and can be retried with `close()`.

Sensitive action values are registered in the shared secret-redaction context
while the body and observers run, including nested sensitive calls. Exported
sensitive schemas also remove `enum` and `pattern` values from reachable definitions;
runtime validation keeps the original constraints.

When a recorder observes a bound action, it brackets dispatch after argument and
binding validation. It writes one outcome for the outer action, including a
`False` return or exception code, not extra rows for primitives or same-extension
actions it calls. Direct calls through a device or its action catalog use that
facade's recording policy, including calls through `device.extensions[name]`.
Calls on the original extension handle use its bound device. Sensitive
arguments are replaced with `***`; results and exception
messages are not persisted. `wait_for_code()` values are tracked in memory so
later actions and logs can mask the code. Verification reads such as `messages()`
also register codes. `device.known_codes_redactor()` returns a live text redactor
for that device and its facades. Full recording screenshots are not redacted.
Register additional exact sensitive values before dispatch with
`device.register_sensitive_values(values)`. The device and its facades retain
them in memory for recording, live observation, and log redaction.

Use `with dispatch_guard(predicate):` to authorize declared actions before
execution, including nested extension actions and actions on unrecorded
devices. The predicate receives `ActionInfo` and denies dispatch by raising.
Nested scopes add checks rather than replace them. `supports_billing_retry`
identifies actions with a declared billing retry path, not permission to spend;
cloud billing policy remains enforced at the client boundary.

Extensions can declare `role: ClassVar[str | None]` (default `None`). The built-in
mailbox and phone-number bases declare `mailbox` and `phone_number`. Device resource
shortcuts select by role, preferring the default `email`/`sms` name, then a unique
match; they never check concrete extension classes.

See [credentials](../credentials/README.md), [proxies](../proxy/README.md), and [devices](../devices/README.md).
