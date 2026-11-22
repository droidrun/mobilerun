# skills/cli/

Exposes commands to list, inspect, install and uninstall bundled skills.

```sh
mobilerun skills list
mobilerun skills install --all --agent codex
```

See [skills](../README.md).

Run `mobilerun skills install` to choose Everything, Bundle (folders), or
Single (flat skills), then an agent. Type to filter the selection, Tab to toggle
items and Enter to install.
In scripts use `install --all`, `install bundle mobile-harness`, or `install single NAME`
with `--agent codex`; both modes also accept `--all`.

Keep skill command presentation here; skill installation logic lives in `skills/`.

## Files

- `__init__.py` — Exports `skills`.
- `common.py` — `options` — shared skill command options.
- `group.py` — `skills` — commands for installing bundled coding-agent skills.
- `install.py` — `install_command`.
- `list.py` — `list_command`.
- `show.py` — `show_command`.
- `uninstall.py` — `uninstall_command`.

**Depends on / used by:** Uses `skills`, `config.settings` and `cli.base`; registered by the main CLI.

`bundle` and `single` are reserved mode names.
