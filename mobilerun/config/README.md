# config/

Runtime settings, user configuration loading, migrations, and secrets access.

**What belongs here / what doesn't:** Put shared configuration and migrations here, not feature implementations or CLI logic.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Package marker |
| `env.py` | Environment-backed secrets access |
| `loader.py` | Typed YAML config files and shared `config.yaml` migrations |
| [migrations/](migrations/README.md) | Versioned configuration migrations |
| `settings.py` | Runtime settings |
| `user_config.py` | Shared user configuration models |

## Quick example

```python
from mobilerun.config.settings import get_settings

timeout = get_settings().MOBILERUN_CLOUD_API_TIMEOUT
print(timeout)
```

Set `MOBILERUN_CLOUD_API_TIMEOUT` in the environment before the call to
override the default. Each `get_settings()` call reads a fresh snapshot.
`ConfigLoader.load()` resolves an explicit path, a scoped override, a nonblank
`MOBILERUN_CONFIG` path, the user config, then defaults. A configured but missing
explicit or environment path is an error. With no user `config.yaml`, the loader
writes defaults. It can also migrate an existing file and write a backup. See
[migrations](migrations/README.md).

**Depends on / used by:** Uses configuration libraries and shared utilities; used across the framework.

## Subclass for your own file

Use `ConfigFile[Model]` for a separate YAML file under
`get_settings().MOBILERUN_USER_CONFIG_DIR`. Specify `CONFIG_FILE` and a no-argument
`config_type`, then implement the two class methods `from_dict(data)` and
`to_dict(config)`. The loader does not depend on a particular model library.

```python
from typing import Any

from pydantic import BaseModel

from mobilerun.config.loader import ConfigFile


class AgentSettings(BaseModel):
    mode: str = "auto"


class AgentConfigFile(ConfigFile[AgentSettings]):
    CONFIG_FILE = "agents.yaml"
    config_type = AgentSettings

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSettings:
        return AgentSettings.model_validate(data)

    @classmethod
    def to_dict(cls, config: AgentSettings) -> dict[str, Any]:
        data = config.model_dump(mode="json", exclude_defaults=True)
        AgentSettings.model_validate(data)
        return data
```

By default, `load()` returns an in-memory default model if the file is missing,
ignores `MOBILERUN_CONFIG`, and neither validates nor adds a `_version` field.
Set `WRITE_DEFAULTS = True` to write on first load or `USE_ENV_PATH = True` to
honour `MOBILERUN_CONFIG` during loads. For migrations, set `CONFIG_VERSION` and
implement `migrate_dict(data)` to return the upgraded mapping with its new
`_version`. Unversioned files start at version 1. A successful upgrade backs
up the original as `<file>.v<old_version>.bak` before replacing it. Override
`parse_yaml(text)` only if the file needs custom YAML parsing.

| Class method | Purpose |
| --- | --- |
| `get_user_config_dir() -> Path` / `get_user_config_path() -> Path` | Locate the user directory or the subclass's file. |
| `load(config_path: str \| None = None) -> Model` | Load and validate the selected YAML file or return defaults. |
| `save(config: Model) -> Path` | Atomically replace the user file. Generic files use `<file>.lock`; `ConfigLoader.save()` keeps its existing unlocked behavior. |
| `update(edit: Callable[[Model], None]) -> Path` | Hold `<file>.lock` across reading, editing in place, and saving. Uses the user path regardless of overrides. A raised edit leaves the YAML untouched. Use this for concurrent read-modify-write callers. |
| `override(config: Model)` | Context manager: within this context, `load()` returns the supplied model for this subclass only. Explicit paths bypass it; it does not change disk writes. |

## Settings overview

`MOBILERUN_CLOUD_NUMBER_READY_TIMEOUT_SECONDS` is the number-activation wait
budget (600 seconds by default; must be positive). Number readiness is
`numbers.retrieve(...).data.state == "active"`, not device readiness or SMS
arrival. Debug logging on `mobilerun.core.phone_number.cloud` shows progress.

All shared settings are read afresh by `get_settings()`. Field names equal their
case-sensitive environment names.

| Section | Environment variables |
| --- | --- |
| Paths | `MOBILERUN_USER_CONFIG_DIR`, `MOBILERUN_CONFIG`, `MOBILERUN_CREDENTIALS_FILE`, `MOBILERUN_MACRO_DIRECTORY` (macro list; defaults to `<MOBILERUN_USER_CONFIG_DIR>/macros`), `MOBILERUN_LOG_LEVEL` |
| Devices | `MOBILERUN_PORTAL_REMOTE_PORT` (default `8080`), `MOBILERUN_ANDROID_DEVICE_TYPE` (AVD name; unset by default), `MOBILERUN_CLOUD_DEVICE_TYPE` (default `android_cloud_phone`) |
| Cloud | `MOBILERUN_BILLING` (`paid` by default; `included` / `no_commerce` restrict charges), `MOBILERUN_CLOUD_PROXY_COUNTRY` (default `us`), `MOBILERUN_CLOUD_PROXY_SMART_IP`, `MOBILERUN_CLOUD_BASE_URL`, `MOBILERUN_CLOUD_API_TIMEOUT`, `MOBILERUN_CLOUD_NUMBER_READY_TIMEOUT_SECONDS`, `MOBILERUN_CLOUD_APP_INSTALL_TIMEOUT_SECONDS`, `MOBILERUN_CLOUD_TASK_TIMEOUT_SECONDS`, `MOBILERUN_CLOUD_TASK_POLL_INTERVAL_SECONDS` |
| Cloud webhooks | `MOBILERUN_WEBHOOK_PAGE_SIZE` (default `100`, from `1` to `100`), `MOBILERUN_WEBHOOK_PING_TIMEOUT` (default `15` seconds, greater than `0` and at most `60`) |
| Device auth | `MOBILERUN_AUTH_URL`, `MOBILERUN_DEVICE_AUTH_REQUEST_TIMEOUT`, `MOBILERUN_DEVICE_AUTH_MIN_POLL_INTERVAL`, `MOBILERUN_DEVICE_AUTH_SLOW_DOWN_BUMP` |
| UI state | `MOBILERUN_STATE_RETRY_DELAYS`, `MOBILERUN_STATE_MAX_RETRIES`, `MOBILERUN_STATE_RECOVERY_AFTER_ATTEMPT`, `MOBILERUN_STATE_MAX_TIMEOUT_ATTEMPTS`, `MOBILERUN_STATE_SLOW_FETCH_NOTICE_S` |
| Screenshots | `MOBILERUN_MODEL_SCREENSHOT_MAX_SIDE` |

`MOBILERUN_AGENTS` overrides the agent registry as a JSON object. Each agent has `skills` and `mcp` user/project path pairs and can supply an `mcp_format` override. Hints can use `{server_name}`. Set an unsupported integration to `null`. MCP-specific environment settings are documented in [mcp/](../mcp/README.md).

Additional non-prefixed settings: `ANDROID_HOME` and `ANDROID_SDK_ROOT` select
Android SDK installations. `MOBILERUN_SMOKE_APK` selects the demo APK.
Phoenix tracing settings are owned by the optional `mobilerun-agents` package.

## Secrets

`get_secrets()` returns a fresh `Secrets` snapshot. Field names match environment
names exactly. Both fields default to `None` and are masked in representations:

- `MOBILERUN_API_KEY` — Cloud API credential.
- `MOBILERUN_DEVICE_TOKEN` — Device authentication token.

Provider and tracing credentials live in `AgentSecrets` under
the `mobilerun-agents` package, not in the base configuration package.

`get_settings().local_credentials_path` resolves `MOBILERUN_CREDENTIALS_FILE`
or `<MOBILERUN_USER_CONFIG_DIR>/credentials.yaml`, expanding `~` in either path.
