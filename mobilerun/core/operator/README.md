# operator/

**Purpose:** The contract for operators that execute device tasks.

**What belongs here / what doesn't:** Put the operator interface here, not concrete agent implementations.

**Contents:**

- `__init__.py` — Operator export.
- `base.py` — Abstract operator interface.

**Examples:**

```python
from mobilerun.core.operator import Operator

# Implement Operator to supply task execution to a device.
```

**Depends on / used by:** Uses shared task result types; used by device implementations and operators.

Operators may override `check(device)` to reject incompatible devices at pairing.
`CloudTaskOperator` requires a cloud client. For local LLM execution, install
`mobilerun-agents` and use its `AgentOperator`.

See [devices](../devices/README.md) and the separate `mobilerun-agents` package.
