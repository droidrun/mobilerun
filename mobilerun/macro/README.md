# macro/

Use `await device.wait(seconds)` instead of `asyncio.sleep()` — plain sleeps are
not recorded by RecordingDevice, so a replayed macro would run without the pause.
Prefer `device.wait` as the default way to pause.

Secret replay always uses the supplied device's credentials providers. Attach
providers with `await provider.bind(device)`; the player never loads its
own credentials. `type_secret` actions reference `package`, `field` and `name`.
Flat `secret`/`secret_id` references become a name lookup without a package filter.
The replay CLI can attach a configured file (`credentials.enabled` and
`credentials.file_path`) or a flat mapping before execution.

**Purpose:** Record device actions and replay them with state guards.

**What belongs here / what doesn't:** Recording, matching, and deterministic replay belong here; LLM fallback belongs in agents/agent_macro.

**Contents:**

- `__init__.py` — Package exports.
- `cli/` — Macro list and replay commands.
- `errors.py` — Replay mismatch errors.
- `models.py` — State normalization and schema types.
- `settings.py` — Environment-backed replay defaults.
- `utils.py` — State matching and logging helpers.
- `operator.py` — Macro-backed operator.
- `player.py` — Replay engine.
- `recorder.py` — Action recording.

## Quick example

Record actions on a local Android emulator and replay the saved macro:

```python
import asyncio, json
from pathlib import Path
from mobilerun.core.devices import AndroidDevice, RecordingDevice
from mobilerun.macro import MacroPlayer
from mobilerun.macro.models import MACRO_SCHEMA_VERSION

async def main():
    device = AndroidDevice("emulator-5554")
    recorded = RecordingDevice(device)
    try:
        await recorded.start_app("com.android.settings")
        await recorded.wait(1)
        await recorded.tap(200, 200)
        path = Path("macro.json")
        path.write_text(json.dumps({"macro_schema_version": MACRO_SCHEMA_VERSION, "actions": recorded.recorded_actions}))
        player = MacroPlayer(device)
        assert await player.replay_macro(player.load_macro_from_file(path))
    finally:
        await recorded.close()

asyncio.run(main())
```

Or replay the saved file from the CLI:

```sh
mobilerun macro replay macro.json emulator-5554
```

See [macro_demo.py](../../tests/demos/macro_demo.py) and [CLI commands](cli/README.md).
Cloud devices can incur charges.

`mobilerun macro list` uses `trajectories` by default. If that default directory
does not exist, it reports no macros (or `[]` with `--json`) and exits successfully.
An explicitly supplied missing directory exits with code 1 and reports
`Macro directory not found: <path>`.

Replay defaults can be overridden with `MOBILERUN_MACRO_DELAY`,
`MOBILERUN_MACRO_STATE_TIMEOUT`, `MOBILERUN_MACRO_STATE_THRESHOLD`, and
`MOBILERUN_MACRO_POLL_INTERVAL`. Set `MOBILERUN_MACRO_DIRECTORY` to change the
default list directory. Explicit replay options take precedence.

**Depends on / used by:** Uses the shared base. Independent of sandbox/, skills/, and agents/; only shared CLI helper `cli/base/` is allowed.
