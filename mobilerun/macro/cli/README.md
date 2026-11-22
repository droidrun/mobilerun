# macro/cli/

Exposes commands to list and replay recorded macros.

```sh
mobilerun macro list
mobilerun macro replay macro.json emulator-5554
```

See [recording and replay](../README.md). Cloud devices can incur charges.

Keep Click commands here; recording and playback implementations live in `macro/`.

## Files

- `__init__.py` — Exports `macro`.
- `group.py` — `macro` — macro commands.
- `list.py` — `list` — list recorded macros.
- `replay.py` — `replay_options`, `replay`, `run_replay` — replay a recorded macro strictly, without agent dependencies.

**Depends on / used by:** Uses `macro`, `cli.base`, device selection, configuration and credentials; registered by the main CLI.
