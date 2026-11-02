---
name: mobilerun-docs
description: Mobilerun documentation reference. Use when users ask about Mobilerun setup, configuration, SDK usage, CLI commands, device setup, agents, architecture, app cards, credentials, tracing, Docker, migration, structured output, or any Mobilerun "how do I..." questions.
---

# Mobilerun

Mobilerun is an open-source (MIT) framework for controlling Android and iOS devices through LLM agents.
It enables mobile automation using natural language commands.

- **GitHub**: https://github.com/droidrun/mobilerun
- **Docs site**: https://docs.mobilerun.ai
- **License**: MIT
- **Install**: `uv tool install mobilerun` (Google Gemini, OpenAI, DeepSeek, Ollama, OpenRouter included by default)
- **Optional extras**: `anthropic`, `langfuse`
- **Requires**: Python 3.11+, ADB, Portal APK on device

## Architecture

Mobilerun uses a multi-agent architecture coordinated by `MobileAgent`:

- **Reasoning mode** (`reasoning=True`): Manager plans → Executor acts → loop until done
- **Direct mode** (`reasoning=False`): CodeActAgent generates and executes Python code directly

Key agents: ManagerAgent (planning), ExecutorAgent (actions), CodeActAgent (direct execution),
ScripterAgent (off-device computation), StructuredOutputAgent (typed data extraction).

Atomic actions available to agents: `click`, `long_press`, `type`, `system_button`, `swipe`,
`open_app`, `get_state`, `take_screenshot`, `complete`.

## Repository Structure

Source code is at `mobilerun/` (Python package). Key locations:

| Path | Description |
|------|-------------|
| `mobilerun/agent/droid/` | MobileAgent orchestrator |
| `mobilerun/agent/codeact/` | CodeActAgent (direct mode) |
| `mobilerun/agent/scripter/` | ScripterAgent (off-device scripts) |
| `mobilerun/agent/oneflows/` | StructuredOutputAgent |
| `mobilerun/agent/utils/` | Tools, executor, tracing, async utils |
| `mobilerun/tools/` | Device tools (ADB, iOS, portal client) |
| `mobilerun/cli/` | CLI entry point (click-based) |
| `mobilerun/config/prompts/` | Jinja2 prompt templates per agent |
| `mobilerun/config/app_cards/` | App-specific instruction cards |
| `mobilerun/credential_manager/` | YAML-based credential storage |
| `mobilerun/telemetry/` | Phoenix tracing integration |

## Documentation

Fetch the relevant page(s) from the docs site based on the user's question. Do not guess — always
read the doc before answering.

| Topic | Page |
|-------|------|
| Overview | https://docs.mobilerun.ai/framework/overview.md |
| Quickstart | https://docs.mobilerun.ai/framework/quickstart.md |
| **Concepts** | |
| Architecture & agents | https://docs.mobilerun.ai/framework/concepts/architecture.md |
| Events & workflows | https://docs.mobilerun.ai/framework/concepts/events-and-workflows.md |
| Prompts | https://docs.mobilerun.ai/framework/concepts/prompts.md |
| Shared state | https://docs.mobilerun.ai/framework/concepts/shared-state.md |
| **Features** | |
| App cards | https://docs.mobilerun.ai/framework/features/app-cards.md |
| Credentials | https://docs.mobilerun.ai/framework/features/credentials.md |
| Custom tools | https://docs.mobilerun.ai/framework/features/custom-tools.md |
| Custom variables | https://docs.mobilerun.ai/framework/features/custom-variables.md |
| Structured output | https://docs.mobilerun.ai/framework/features/structured-output.md |
| Telemetry | https://docs.mobilerun.ai/framework/features/telemetry.md |
| Tracing | https://docs.mobilerun.ai/framework/features/tracing.md |
| Vision | https://docs.mobilerun.ai/framework/features/vision.md |
| **Guides** | |
| CLI usage | https://docs.mobilerun.ai/framework/guides/cli.md |
| Device setup | https://docs.mobilerun.ai/framework/guides/device-setup.md |
| Docker | https://docs.mobilerun.ai/framework/guides/docker.md |
| **SDK** | |
| MobileAgent | https://docs.mobilerun.ai/framework/sdk/droid-agent.md |
| ADB tools | https://docs.mobilerun.ai/framework/sdk/adb-tools.md |
| iOS tools | https://docs.mobilerun.ai/framework/sdk/ios-tools.md |
| Base tools | https://docs.mobilerun.ai/framework/sdk/base-tools.md |
| Configuration | https://docs.mobilerun.ai/framework/sdk/configuration.md |
| API reference | https://docs.mobilerun.ai/framework/sdk/reference.md |

For deeper implementation details, check the source code directly in `mobilerun/`.
