# config/

Runtime settings, user configuration loading, migrations, and secrets access.

**What belongs here / what doesn't:** Put shared configuration and migrations here, not feature implementations or CLI logic.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Package marker |
| `env.py` | Environment-backed secrets access |
| `loader.py` | User config loading and migration |
| `paths.py`, `prompts.py` | Package asset resolution and prompt loading |
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
`ConfigLoader.load()` resolves an explicit path, an existing `MOBILERUN_CONFIG`
path, the user config, then package defaults. It can create a user config or
migrate an existing file and write a backup; use it when those writes are intended.
See [migrations](migrations/README.md).

**Depends on / used by:** Uses configuration libraries and shared utilities; used across the framework.

## Settings overview

`MOBILERUN_CLOUD_NUMBER_READY_TIMEOUT_SECONDS` is the number-activation wait
budget (600 seconds by default; must be positive). Number readiness is
`numbers.retrieve(...).data.state == "active"`, not device readiness or SMS
arrival. Debug logging on `mobilerun.core.phone_number.cloud` shows progress.

All shared settings are read afresh by `get_settings()`. Field names equal their
case-sensitive environment names.

| Section | Environment variables |
| --- | --- |
| Paths | `MOBILERUN_USER_CONFIG_DIR`, `MOBILERUN_CONFIG`, `MOBILERUN_CREDENTIALS_FILE`, `MOBILERUN_LOG_LEVEL` |
| Devices | `MOBILERUN_PORTAL_REMOTE_PORT` (default `8080`), `MOBILERUN_ANDROID_DEVICE_TYPE` (AVD name; unset by default), `MOBILERUN_CLOUD_DEVICE_TYPE` (default `android_cloud_phone`) |
| Cloud | `MOBILERUN_BILLING` (`paid` by default; `included` / `no_commerce` restrict charges), `MOBILERUN_CLOUD_PROXY_COUNTRY` (default `us`), `MOBILERUN_CLOUD_PROXY_SMART_IP`, `MOBILERUN_CLOUD_BASE_URL`, `MOBILERUN_CLOUD_API_TIMEOUT`, `MOBILERUN_CLOUD_NUMBER_READY_TIMEOUT_SECONDS`, `MOBILERUN_CLOUD_APP_INSTALL_TIMEOUT_SECONDS`, `MOBILERUN_CLOUD_TASK_TIMEOUT_SECONDS`, `MOBILERUN_CLOUD_TASK_POLL_INTERVAL_SECONDS` |
| Device auth | `MOBILERUN_AUTH_URL`, `MOBILERUN_DEVICE_AUTH_REQUEST_TIMEOUT`, `MOBILERUN_DEVICE_AUTH_MIN_POLL_INTERVAL`, `MOBILERUN_DEVICE_AUTH_SLOW_DOWN_BUMP` |
| UI state | `MOBILERUN_STATE_RETRY_DELAYS`, `MOBILERUN_STATE_MAX_RETRIES`, `MOBILERUN_STATE_RECOVERY_AFTER_ATTEMPT`, `MOBILERUN_STATE_MAX_TIMEOUT_ATTEMPTS`, `MOBILERUN_STATE_SLOW_FETCH_NOTICE_S` |
| Screenshots | `MOBILERUN_MODEL_SCREENSHOT_MAX_SIDE` |

`MOBILERUN_AGENTS` overrides the agent registry as a JSON object. Each agent has `skills` and `mcp` user/project path pairs and an `mcp_format` describing its server key, transport, API-key reference, and optional `oauth_sign_in` hint. Hints can use `{server_name}`. Set an unsupported integration to `null`. MCP-specific environment settings are documented in [mcp/](../mcp/README.md).

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
