# Macros

Build a replayable macro from a device [run](../core/runs/README.md). Each
successful selected action becomes a JSONL row with its original step ID,
arguments, and any before/after screen guards. Failed actions and actions
with masked arguments are omitted. Run timestamps, actors, errors, and
screenshots are not part of the macro.

```python
from pathlib import Path

from mobilerun.macro import MacroPlayer, build_macro, load_macro

path = build_macro(Path("/path/to/run/steps.jsonl"), "open-settings")

async def replay(device):
    player = MacroPlayer(device)
    await player.replay_macro(load_macro(path))
```

Names and directories load the latest numeric version. Pass an exact file to
pin a version. For example, `open-settings` can load `v10.jsonl` while
`macros/open-settings/v9.jsonl` still pins version 9. Publishing never replaces
an existing version. Legacy v2 `macro.json` files are not supported.

By default the macro store is `<MOBILERUN_USER_CONFIG_DIR>/macros`.
Set `MOBILERUN_MACRO_DIRECTORY` to change it, or use `mobilerun macro path` to
find its current location. The replay delay, guard timeout, similarity
threshold and polling interval default to 1 second, 5 seconds, 0.85 and 0.5
seconds, respectively. Explicit `MacroPlayer` options take precedence over the
corresponding `MOBILERUN_MACRO_*` settings.

The player uses recorded coordinates, even if an element moves. It does not
open, wrap, or close the device. A step with no guard skips that check; a
present guard requires screen evidence. `MacroMismatch` identifies the
original row and failing phase (`before`, `action`, or `after`). A mismatch
during masked credential replay stops without handing off the device.

Credentials come only from providers bound to the supplied device. Replay
checks the selected steps before dispatch; protected credentials and masked
literals are not replayed. The [CLI](cli/README.md) also asks for consent
before device selection if the steps may send SMS. Cloud use can incur charges.

| File | Responsibility |
| --- | --- |
| `builder.py`, `models.py` | Build, version and load typed macro rows |
| `player.py`, `errors.py` | Guarded replay and phase-specific failures |
| `operator.py` | Run a macro through the shared device operator interface |
| `settings.py`, `utils.py` | Replay defaults, fingerprint matching and logging |
| `cli/` | List and replay macros from the terminal |

Agent-assisted repair belongs to the separate `mobilerun_agents` add-on.
