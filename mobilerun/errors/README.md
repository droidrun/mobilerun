# errors/

Shared exception types for device, cloud, credentials, and other framework failures.

**What belongs here / what doesn't:** Put reusable error definitions here, not recovery workflows or CLI output.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py`, `base.py` | Public exports and shared error hierarchy |
| `cloud.py` | Cloud API, authentication, billing and proxy failures |
| `credentials.py` | Credential lookup, file and policy failures |
| `device.py` | Device authentication, selection, Portal and state failures |
| `operators.py` | Operator, extension, expectation and placeholder failures |

## Quick example

```python
from mobilerun.errors import CloudError, CredentialError, DeviceError, MobilerunError

def run_safely(operation):
    try:
        return operation()
    except (CloudError, CredentialError, DeviceError) as exc:
        print(f"Operation failed: {type(exc).__name__}")
    except MobilerunError as exc:
        print(f"Other Mobilerun error: {type(exc).__name__}")
```

See [diagnostics](../diagnostics/README.md).

**Depends on / used by:** Uses standard exception types; used across shared packages and features.

## Cloud error safety

Construct cloud errors with keyword diagnostic fields, for example
`CloudServiceUnavailable(operation="List SMS messages", status=503)`.
These instances set `value_safe=True` and render fixed message templates, known
operation labels, validated HTTP status codes and support request IDs. Raw
`status`, `detail` and other diagnostic attributes remain available for inspection;
they are not arbitrary message text. Do not put credentials in reference fields.

Legacy positional messages remain accepted for compatibility, but set instance
`value_safe=False`. An extension action wraps them instead of passing their text
through. Use `code` for the supported fixed remediation templates.

When an action propagates a safe typed error, it retains the same error object and
attributes but removes implicit contexts and unsafe explicit causes. Only acyclic,
value-safe explicit causes remain. This applies at INFO and DEBUG, including
sensitive and nested action scopes. All notes are removed from every retained
exception: free-text notes do not inherit the error's safety classification.

Stock cloud errors render a private, write-once message rather than mutable
`.args`. The action boundary revalidates rendering and rejects changed `.args`.
Every unaudited class in a cloud error's MRO must contain only plain constants
(strings, numbers, booleans, `None`, or recursively constant tuples/frozensets)
and standard class bookkeeping. Methods and descriptors are rejected by default,
including properties, lookup hooks and unrelated helper methods. Hint rendering
reads raw static class data; it never invokes an instance property or lookup hook.

Classes with additional behavior must each deliberately declare
`safe_rendering: ClassVar[bool] = True`; inheriting the flag does not audit new
behavior, and a child opt-in does not audit an unsafe parent or mixin. Even opted-in
subclasses must render the stored message exactly. This is an explicit author
promise, not an automatic audit of custom code.
Rejected rendering contracts are wrapped without a cause, including at DEBUG.

Direct client calls outside an action retain
their original transport chains for diagnostics.
