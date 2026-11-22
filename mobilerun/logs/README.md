# logs/

Shared logging handlers, action logging, and progress output.

**What belongs here / what doesn't:** Put logging configuration and handlers here, not command or feature logic.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Package marker |
| `actions.py` | Action logging helpers |
| `handlers.py` | Logging setup and CLI/TUI handlers |
| `progress.py` | Progress output to stderr |

## Quick example

```python
from mobilerun.logs.handlers import CLILogHandler, configure_logging

configure_logging(debug=True, handler=CLILogHandler())
```

This replaces handlers on the `mobilerun` logger and sets its level to DEBUG.
See [configuration](../config/README.md) and [CLI commands](../cli/README.md).

Agent logging supports `quiet`, `normal` (default), `verbose`, and `debug`.
Set `logging.level` in the configuration file or `MOBILERUN_LOG_LEVEL` in
the environment. For `agent run`, `--quiet`, `--verbose`, `--debug`, and
`--no-debug` override both. The legacy `logging.debug: true` selects debug
when no environment or CLI level overrides it.
`-q` is an alias for `--quiet`; `--verbose` has no short alias because `-v`
continues to mean `--remote`.

**Depends on / used by:** Uses Python logging and presentation dependencies; used by CLI and framework code.
