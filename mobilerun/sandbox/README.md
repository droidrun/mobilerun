# sandbox/

**Purpose:** Define sandbox apps, routines, and tasks. Task and client calls
raise `SandboxNotImplemented` until sandbox-api ships. There is no registered
`mobilerun sandbox` CLI command.

**What belongs here / what doesn't:** Sandbox script APIs belong here; device and agent implementations do not.

**Contents:**

- `__init__.py` — Package exports.
- `app.py` — App and task registration.
- `models.py` — Task wrapper and routine decorator.
- `client.py` — Deployed script and run result clients.
- `errors.py` — Sandbox errors.

**Examples:**

```python
from mobilerun.sandbox import App

app = App("example")

@app.task()
def greet():
    return "hello"
```

**Depends on / used by:** Uses the shared base. Independent of macro/, skills/, and agents/; only shared CLI helper `cli/base/` is allowed.
