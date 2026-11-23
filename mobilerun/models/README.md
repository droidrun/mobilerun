# models/

Core's models. Feature packages keep their own models inside their folder.

Defines transport-independent values for apps, billing, credentials and task results.

Keep shared value types here; device lifecycle and API calls belong in `core/` and `clients/`.

## Files

- `__init__.py` — Exports shared value types, including `Location`, `App`, `Billing`, `Credential`, and task results.
- `apps.py` — `App` — application value type shared by devices and clients.
- `billing.py` — `Billing`, `Plan` — billing modes shared by configuration and core resources.
- `credentials.py` — `Credential`, `CredentialMetadata` — value-safe credential data and metadata.
- `checkout.py` — `Checkout` and `PaymentSnapshot` — checkout and payment values.
- `location.py` — `Location` — latitude and longitude reported by a device.
- `proxy.py` — `ProxyStatus` — proxy connection status.
- `recording.py` — `RecordedAction` — recorded device action data.
- `ui.py` — `UIElement` — UI snapshot element data.
- `results.py` — `ExecutionResult`, `ExpectationVerdict`, `ExpectationResult` — typed task results, independent of execution and transport.

```python
from mobilerun.models import Billing

billing = Billing.INCLUDED
```

See [configuration](../config/README.md).

**Depends on / used by:** Python dataclasses, enums and Pydantic; used by device, cloud-client and agent code.
