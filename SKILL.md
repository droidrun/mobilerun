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
- **Install**: `uv tool install mobilerun` (Google Gemini, OpenAI, Ollama, OpenRouter included by default)
- **Optional extras**: `anthropic`, `deepseek`, `langfuse`
- **Requires**: Python 3.11+, ADB, Portal APK on device

## Architecture

Mobilerun uses a multi-agent architecture coordinated by `MobileAgent`:

- **Reasoning mode** (`reasoning=True`): Manager plans → Executor acts → loop until done
- **Direct mode** (`reasoning=False`): CodeActAgent generates and executes Python code directly

Key agents: ManagerAgent (planning), ExecutorAgent (actions), CodeActAgent (direct execution),
ScripterAgent (off-device computation), StructuredOutputAgent (typed data extraction).

Atomic actions available to agents: `click`, `long_press`, `type`, `system_button`, `swipe`,
`open_app`, `get_state`, `take_screenshot`, `remember`, `complete`.

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

Read the relevant file(s) from `docs/v5/` based on the user's question. Do not guess — always
read the doc before answering.

| Topic | File |
|-------|------|
| Overview | docs/v5/overview.mdx |
| Quickstart | docs/v5/quickstart.mdx |
| **Concepts** | |
| Architecture & agents | docs/v5/concepts/architecture.mdx |
| Events & workflows | docs/v5/concepts/events-and-workflows.mdx |
| Prompts | docs/v5/concepts/prompts.mdx |
| Scripter agent | docs/v5/concepts/scripter-agent.mdx |
| Shared state | docs/v5/concepts/shared-state.mdx |
| **Features** | |
| App cards | docs/v5/features/app-cards.mdx |
| Credentials | docs/v5/features/credentials.mdx |
| Custom tools | docs/v5/features/custom-tools.mdx |
| Custom variables | docs/v5/features/custom-variables.mdx |
| Structured output | docs/v5/features/structured-output.mdx |
| Telemetry | docs/v5/features/telemetry.mdx |
| Tracing | docs/v5/features/tracing.mdx |
| **Guides** | |
| CLI usage | docs/v5/guides/cli.mdx |
| Device setup | docs/v5/guides/device-setup.mdx |
| Docker | docs/v5/guides/docker.mdx |
| Migration v3→v4 | docs/v5/guides/migration-v3-to-v4.mdx |
| **SDK** | |
| MobileAgent | docs/v5/sdk/droid-agent.mdx |
| ADB tools | docs/v5/sdk/adb-tools.mdx |
| iOS tools | docs/v5/sdk/ios-tools.mdx |
| Base tools | docs/v5/sdk/base-tools.mdx |
| Configuration | docs/v5/sdk/configuration.mdx |
| API reference | docs/v5/sdk/reference.mdx |

For deeper implementation details, check the source code directly in `mobilerun/`.
