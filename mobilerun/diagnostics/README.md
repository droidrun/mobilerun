# diagnostics/

Reusable health checks and typed diagnostic results for the CLI doctor.

**What belongs here / what doesn't:** Put reusable environment and device checks here, not CLI command wiring.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Package marker |
| `checks.py` | SDK, config, device, and Portal checks |
| `results.py` | Check result and status types |
| `versions.py` | SDK and Portal version helpers |

## Quick example

```sh
mobilerun doctor
```

Use `mobilerun doctor --fix` only when you want it to repair Portal installation
and accessibility settings. Cloud device use can incur charges.

For a typed result in Python:

```python
from mobilerun.diagnostics.results import CheckResult, Status

result = CheckResult(name="config", status=Status.PASS, message="Ready")
```

See the [CLI overview](../cli/README.md).

**Depends on / used by:** Uses shared config, clients, and device dependencies; used by the doctor command.
