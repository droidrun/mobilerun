# Optional add-ons

The base package discovers add-ons using installed distribution entry points.
Discovery reads metadata without importing the add-on. First-party macro, mcp,
and skills commands use the same mechanism in the `mobilerun` distribution.
Sandbox currently has no CLI group.

## Write a plugin

Declare entry points in your add-on's `pyproject.toml`:

```toml
[project.entry-points."mobilerun.cli"]
example = "example_plugin.cli:command"

[project.entry-points."mobilerun.operators"]
agent_macro = "example_plugin.replay:handoff"
```

- `mobilerun.cli`: each value resolves to a `click.Command` or `click.Group`.
  Its name appears in root help. The implementation is imported when invoked,
  not when help is listed.
- `mobilerun.operators`: `agent_macro` resolves to a **plain callable**, not a
  Click-decorated callback. The base invokes it with keyword arguments
  `provider: str | None`, `model: str | None`, `path: str`, a resolved
  `target: DeviceTarget`, `delay: float`, `start_from: int`,
  `max_steps: int | None`, `debug: bool`, `dry_run: bool`,
  `state_timeout: float`, `state_threshold: float`, and
  `config_path: str | None`. It owns configuration and replay. Do not reparse
  device flags: `target` is already resolved. The callable loads only when a
  replay requests agent fallback.

Built-in command and alias names win over add-on CLI names. For duplicate plugin names, a single entry from the `mobilerun` distribution
wins over third-party entries. Other ambiguous duplicates are skipped with a
warning, rather than selected arbitrarily.
`mobilerun plugins list` shows each entry point, distribution and version;
`--json` returns machine-readable rows. The default status is `not checked`;
`--check` loads unshadowed entries and reports `ok` or `broken` (without
printing the import exception). Shadowed entries are reported as `shadowed`.
Loading during `--check` can execute plugin import code.

Test your installed distribution without devices or network calls. For example,
with a fake `run_replay` and configuration loader:

```python
from importlib.metadata import entry_points

def test_handoff(monkeypatch):
    from example_plugin import replay
    calls = []
    monkeypatch.setattr(replay, "run_replay", lambda *a, **kw: calls.append((a, kw)))
    # Also replace the configuration loader with a fake.
    ep = next(ep for ep in entry_points(group="mobilerun.operators")
              if ep.name == "agent_macro")
    ep.load()(provider=None, model=None, path="macro.json", target=object(),
              delay=0.0, start_from=1, max_steps=None, debug=False,
              dry_run=False, state_timeout=1.0, state_threshold=0.5,
              config_path=None)
    assert calls[0][1]["path"] == "macro.json"
```

The `mobilerun/cli/plugins/` package contains the top-level CLI for this
core-side concern; `mobilerun/addons/` owns discovery and shared contracts.
See the [package map](../README.md) and [CLI](../cli/README.md).
