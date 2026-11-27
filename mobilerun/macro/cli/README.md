# macro/cli/

Exposes commands to list and replay recorded macros.

```sh
mobilerun macro list
mobilerun macro replay open-settings --dry-run
mobilerun macro replay open-settings emulator-5554
mobilerun macro replay ~/.config/mobilerun/macros/open-settings/v1.jsonl emulator-5554
```

Names and folders select the latest numeric version. An exact JSONL file pins
that version. Use `mobilerun macro path` to locate your macro store. `--start-from`
selects a one-based row position, while preview and mismatch output show the
original recorded step ID. `--dry-run` previews the selected actions without a
device and omits private arguments and screen evidence. Live replay asks for
consent before selecting a device if the selected steps may send SMS; use
`--yes` only to approve those potential charges.

See [recording and replay](../README.md). Cloud devices can incur charges.

Keep Click commands here; recording and playback implementations live in `macro/`.

## Files

- `__init__.py` — Exports `macro`.
- `group.py` — `macro` — macro commands.
- `list.py` — `list` — list recorded macros.
- `replay.py` — `replay_options`, `replay`, `run_replay` — replay a recorded macro strictly, without agent dependencies.

**Depends on / used by:** Uses `macro`, `cli.base`, device selection, configuration and credentials; registered by the main CLI.
