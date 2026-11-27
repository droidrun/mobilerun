# config/migrations/

Upgrades older saved configuration schemas to current versions.

Keep versioned migration functions and their runner here; runtime settings belong in `config/settings.py`.

## Quick example

```python
from mobilerun.config.migrations import CURRENT_VERSION, migrate

config = migrate({"_version": CURRENT_VERSION})
print(config["_version"])
```

`migrate` transforms the supplied mapping; the config loader handles any file
backup and persistence. See [configuration](../README.md).

## Files

| Path | Purpose |
| --- | --- |
| `__init__.py`, `runner.py` | Export, discover and run migrations |
| `v002_add_code_exec.py` | Rename codeact config and add codeact flag |
| `v003_add_auto_setup.py` | Add device auto-setup |
| `v004_remove_deprecated_agents.py` | Remove deprecated agent configs and rename prompt paths |
| `v005_remove_external_agents.py` | Remove legacy external agent configs |
| `v006_disabled_tools_default_sentinel.py` | Use the disabled-tools default sentinel |
| `v007_manager_prompt_auto.py` | Select manager prompt automatically |
| `v008_bundled_assets.py` | Relocate bundled asset references |

**Depends on / used by:** Uses Python import machinery; called by the configuration loader.
