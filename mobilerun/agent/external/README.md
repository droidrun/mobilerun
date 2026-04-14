# External Agents

External agents are self-contained modules that receive raw ADB access and run independently from Droidrun's internal tools.

## Quick Start

1. Add your agent as a file or folder in this directory
2. Configure it in `config.yaml` under `external_agents.<name>`
3. Run with `droidrun run "task" --agent <name>`

## Agent Contract

Your agent must expose an async `run()` function:

```python
from async_adbutils import AdbDevice

async def run(
    device: AdbDevice,
    instruction: str,
    config: dict,
    max_steps: int,
) -> dict:
    ...
    return {"success": True, "reason": "Task completed", "steps": 5}
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `device` | `AdbDevice` | Raw ADB connection, already connected. From `async_adbutils`. |
| `instruction` | `str` | The user's task (e.g., "open settings and enable dark mode"). |
| `config` | `dict` | Your agent's config from `external_agents.<name>` in config.yaml. |
| `max_steps` | `int` | Maximum number of steps allowed. |

### Return Value

A dict with:

| Key | Type | Description |
|-----|------|-------------|
| `success` | `bool` | Whether the task completed successfully. |
| `reason` | `str` | Human-readable result or failure reason. |
| `steps` | `int` | Number of steps taken. |

## AdbDevice

`AdbDevice` from `async_adbutils` provides raw ADB access. Every method is a thin wrapper over `adb shell`:

```python
# Tap
await device.click(500, 500)  # or: device.shell("input tap 500 500")

# Swipe
await device.swipe(100, 500, 100, 200, duration=0.3)

# Screenshot (raw PNG bytes)
png_bytes = await device.screenshot_bytes()

# UI hierarchy (raw uiautomator XML)
xml = await device.dump_hierarchy()

# Key events
await device.keyevent(4)  # BACK

# Type text
await device.send_keys("hello")

# Run any shell command
output = await device.shell("dumpsys window")

# Screen size
w, h = await device.window_size()

# App management
await device.app_start("com.example.app")
await device.app_stop("com.example.app")
packages = await device.list_packages()

# Current app
info = await device.app_current()  # -> RunningAppInfo(package, activity, pid)
```

## File Structure

An agent can be a single file or a package:

```
# Single file
external/my_agent.py

# Package (for larger agents)
external/my_agent/
    __init__.py      # must expose run() and optionally DEFAULT_CONFIG
    prompts.py
    parser.py
    actions.py
```

## Configuration

Agent config goes in `config.yaml`:

```yaml
external_agents:
  my_agent:
    api_key: "sk-..."
    model: "model-name"
    base_url: "http://localhost:8000/v1"
    # any other settings your agent needs
```

The entire `external_agents.my_agent` dict is passed as the `config` parameter to `run()`.

Optionally, your module can define `DEFAULT_CONFIG` — a dict of defaults that gets merged under the user's config:

```python
DEFAULT_CONFIG = {
    "temperature": 0.0,
    "history_n": 3,
}
```

## Rules

- **Zero imports from `droidrun`** — your agent must be fully self-contained
- Bring your own LLM client (`openai`, `httpx`, `anthropic`, etc.)
- Bring your own prompts, parsing, and action logic
- Only use `device` for all device interaction — no Portal, no internal tools
- The `device` is already connected — do not call `connect()` on it
