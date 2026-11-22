# models/

Defines transport-independent values for apps, billing, credentials and task results.

Keep shared value types here; device lifecycle and API calls belong in `core/` and `clients/`.

## Files

- `__init__.py` — Exports `App`, `Billing`, `Plan`, `Credential`, `ExecutionResult`, `ExpectationResult`, `ExpectationVerdict`.
- `apps.py` — `App` — application value type shared by devices and clients.
- `billing.py` — `Billing`, `Plan` — billing modes shared by configuration and core resources.
- `credentials.py` — `Credential` — value-safe credential data.
- `results.py` — `ExecutionResult`, `ExpectationVerdict`, `ExpectationResult` — typed task results, independent of execution and transport.

```python
from mobilerun.models import Billing

billing = Billing.INCLUDED
```

See [configuration](../config/README.md).

**Depends on / used by:** Python dataclasses, enums and Pydantic; used by device, cloud-client and agent code.
