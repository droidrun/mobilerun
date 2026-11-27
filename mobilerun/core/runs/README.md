# runs/

Store device-action outcomes independently of macros or agents. The writer
manages lazy creation, status and step attribution. Callers appending `RunStep`
directly must redact their own arguments and evidence.

```python
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from mobilerun.core.runs import RunWriter
from mobilerun.models import DeviceInfo, RunRecord, RunStep

now = datetime.now(timezone.utc)
writer = RunWriter(
    RunRecord(
        run_id=uuid4().hex, kind="recording", started_at=now,
        device=DeviceInfo(platform="android", identifier="local"),
    ),
    Path("runs"),
)
writer.append(RunStep(i=1, started_at=now, finished_at=now, action="tap",
                      args={"x": 10, "y": 20}, ok=True, by="script"))
writer.finish("succeeded")
```

`run_dir` is the prospective path. It is created on the first append, or
explicitly by `start()` when another evidence sink needs it first. Finishing
an unused writer creates no files. Each append flushes one JSONL row before
advancing `step_index`; `finish()` replaces `run.json` atomically. No resume or
crash durability is provided. After an append I/O failure, do not rerun the
device action or append again. The writer may still finalize a started run.

Use `recording_scope(enabled=False)` to suppress recording in the current task,
or `recording_scope(by="agent")` to override the actor. `recording_policy()`
returns the effective `(enabled, actor_override)`. A nested scope cannot lift
suppression from an outer scope.

`collect_steps()` includes committed rows from the current context and tasks
spawned inside the block. Await those tasks before leaving the block. Nested
blocks see their own rows and contribute them to enclosing blocks.

Each step has UTC action start and finish times, excluding evidence capture.
`RecordingDevice(capture="full")` (the default) stores structural before/after
fingerprints, redacted UI and context JSON, and PNG files when available.
Snapshots live under `screens/<i>.<before|after>.*`; a step can reference the
previous step's after files as its before state. `capture="actions"` skips
all automatic screen observation and stores only the action and outcome data.
**Screenshots are not redacted. Use `capture="actions"` for sensitive flows.**

| File | Purpose |
| --- | --- |
| `writer.py` | Append action rows, publish metadata and collect committed step IDs |
| `session.py` | Coordinate dispatch, extension observation and capture |
| `capture.py` | Observe, redact and reuse before/after screen evidence |
| `__init__.py` | Public run API |

See [run values](../../models/recording.py) and [run settings](../../config/README.md#settings-overview).
