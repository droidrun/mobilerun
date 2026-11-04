<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="./static/mobilerun-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/mobilerun.png">
  <img src="./static/mobilerun.png"  width="full">
</picture>

<p align="center">
  <strong>Mobilerun is an open-source framework for mobile app testing, QA, and running AI agents on real Android and iOS devices.</strong><br>
  It gives agents mobile-native tools to inspect UI state, understand screenshots, tap, swipe, type, plan multi-step workflows, and return results through a CLI or Python API.
</p>

<div align="center">

<a href="https://docs.mobilerun.ai">📕 Documentation</a>
·
<a href="https://cloud.mobilerun.ai">☁️ Mobilerun Cloud</a>
·
<a href="https://github.com/droidrun/mobile-harness">🛡️ Mobile Harness</a>

[![GitHub stars](https://img.shields.io/github/stars/droidrun/mobilerun?style=social)](https://github.com/droidrun/mobilerun/stargazers)
[![mobilerun.ai](https://img.shields.io/badge/mobilerun.ai-white)](https://mobilerun.ai)
[![Twitter Follow](https://img.shields.io/twitter/follow/mobilerun_ai?style=social)](https://x.com/mobilerun_ai)
[![Discord](https://img.shields.io/discord/1360219330318696488?color=white&label=Discord&logo=discord&logoColor=white)](https://discord.gg/ZZbKEZZkwK)
[![Telegram Support](https://img.shields.io/badge/Telegram-Support-26A5E4?logo=telegram&logoColor=white)](https://t.me/+_r43WkbbyiA1OGUy)
[![Benchmark](https://img.shields.io/badge/Benchmark-91.4﹪-white)](https://mobilerun.ai/benchmark)



<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=983810&theme=dark&period=daily&t=1753948032207">
  <source media="(prefers-color-scheme: light)" srcset="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=983810&theme=neutral&period=daily&t=1753948125523">
  <a href="https://www.producthunt.com/products/droidrun-framework-for-mobile-agent?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-droidrun" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=983810&theme=neutral&period=daily&t=1753948125523" alt="Droidrun - Give&#0032;AI&#0032;native&#0032;control&#0032;of&#0032;physical&#0032;&#0038;&#0032;virtual&#0032;phones&#0046; | Product Hunt" style="width: 200px; height: 54px;" width="200" height="54" /></a>
</picture>


[Deutsch](https://zdoc.app/de/droidrun/mobilerun) | 
[Español](https://zdoc.app/es/droidrun/mobilerun) | 
[français](https://zdoc.app/fr/droidrun/mobilerun) | 
[日本語](https://zdoc.app/ja/droidrun/mobilerun) | 
[한국어](https://zdoc.app/ko/droidrun/mobilerun) | 
[Português](https://zdoc.app/pt/droidrun/mobilerun) | 
[Русский](https://zdoc.app/ru/droidrun/mobilerun) | 
[中文](https://zdoc.app/zh/droidrun/mobilerun)

</div>


<p align="center">
  <img src="./static/mobilerun-demo.gif" alt="Mobilerun automating a phone with natural language" width="320">
</p>

- 🤖 Control Android and iOS devices with natural language commands
- 🔀 Use OpenAI, Anthropic, Gemini, xAI, Ollama, DeepSeek, OpenRouter, and OpenAI-compatible models
- 🧠 Run direct tasks or enable reasoning mode for complex multi-step automation
- 💻 Automate from the CLI, a terminal UI, Docker, or Python code
- 🐍 Extend agents with custom tools, structured output, app cards, and credentials
- 📸 Combine accessibility trees with screenshots for visual understanding
- 🫆 Trace execution with Arize Phoenix or Langfuse

Use the framework when you want to run the agent on your machine. Use [Mobilerun Cloud](https://cloud.mobilerun.ai) when you want a ready-to-go solution for your local phones or cloud-hosted virtual/physical phones, managed infrastructure, and API-driven device workflows without running the agent on your local machine. [Check out our benchmark results](https://mobilerun.ai/benchmark).

## 🎯 Platform Support

| Target | Supported | Setup |
| --- | :---: | --- |
| Android Real Device | ✅ | `adb` or the [Portal](https://github.com/droidrun/mobilerun-portal) app |
| Android Emulator | ✅ | Running emulator (`adb`) |
| iOS Real Device | ✅ | [`mobilerun-ios`](https://docs.mobilerun.ai/framework/guides/device-setup) over USB |
| iOS Simulator | ✅ | [iOS Portal](https://github.com/droidrun/ios-portal) app |
| Android on [Mobilerun Cloud](https://mobilerun.ai) | ✅ | Hosted phones or your own |
| iOS on [Mobilerun Cloud](https://mobilerun.ai) | ✅ | Hosted iPhones or your own |

## 📦 Installation

> **Note:** Python 3.14 is not currently supported. Please use Python `>=3.11,<3.14`.

Install Mobilerun with [`uv`](https://docs.astral.sh/uv/):

```bash
# CLI usage
uv tool install mobilerun
```

```bash
# CLI + Python integration
uv pip install mobilerun
```

Most LLM providers are included by default. For Anthropic support, install the optional extra:

```bash
uv tool install "mobilerun[anthropic]"
```

## 🚀 Quickstart

```bash
uv tool install mobilerun
mobilerun setup
mobilerun configure
mobilerun run "Open settings and turn on dark mode"
```

Before starting, make sure you have [ADB](https://developer.android.com/studio/releases/platform-tools) installed and an Android device with Developer options and USB debugging enabled. For iOS, set up [`mobilerun-ios`](https://docs.mobilerun.ai/framework/guides/device-setup) for an iPhone or the [iOS Portal](https://github.com/droidrun/ios-portal) app for the simulator.

### 1. Install the Portal on your device

```bash
mobilerun setup
```

This installs the Mobilerun Portal app, enables its accessibility service, and prepares the device for local control.

### 2. Verify the connection

```bash
mobilerun ping
```

You should see confirmation that the Portal is installed and accessible.

### 3. Configure your LLM provider

```bash
mobilerun configure
```

The wizard walks you through choosing a provider, auth method, and model. You can also use provider environment variables such as `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, or `MINIMAX_API_KEY`.

### 4. Run your first command

```bash
mobilerun run "Open the settings app and tell me the Android version"
```

Useful run options:

```bash
mobilerun run "Open settings and turn on dark mode"
mobilerun run "What app is currently open?" --vision
mobilerun run "Find a contact named John and send him an email" --reasoning
mobilerun run "Take a screenshot" --ios
mobilerun run "Open Settings" --steps 30 --debug
```

Read the full [framework documentation](https://docs.mobilerun.ai/framework/quickstart).

### Custom action selection

ExecutorAgent and FastAgent take an `action_selector`, not an `llm`.
Legacy `llm=` keyword calls remain supported through `**kwargs`: the LLM is
wrapped in `LlamaIndexLLMSelector` and a DEBUG deprecation message is logged.
An explicit `action_selector` takes precedence if both are supplied.
MobileAgent accepts one selector through `action_selectors=selector`, or
`action_selectors={"fast_agent": selector, "executor": selector}` for per-role
overrides. The active role is `fast_agent` without reasoning and
`executor` with reasoning. Supplied selectors are preserved; missing active roles
are loaded from config (decision profiles take precedence over LLM profiles).
Planning and helper LLMs still load from `config.llm_profiles`.

An explicitly supplied `action_selectors=LlamaIndexLLMSelector(llm)` reuses that
same native LLM for planning, execution, app opening, and structured output,
matching the previous single-LLM input. Other selectors use config profiles
for any planning and helper LLMs.

Legacy `MobileAgent(llms=llm)` and `MobileAgent(llms={...}, config=config)`
remain supported with a deprecation warning. The active action LLM is wrapped
in a `LlamaIndexLLMSelector`; per-role assignments and config merging are preserved.
If both are supplied, `action_selectors` takes precedence and a warning reports
that the legacy `llms` argument was ignored.

```python
from mobilerun import MobileAgent
from mobilerun.agent.selectors import (
    DecisionSelector,
    FallbackSelector,
    LlamaIndexLLMSelector,
)

agent = MobileAgent(
    "Open settings",
    config=config,
    action_selectors={
        "fast_agent": FallbackSelector(
            primary=DecisionSelector(decision_model),
            fallbacks=[LlamaIndexLLMSelector(llm)],
        ),
        "executor": DecisionSelector(decision_model),
    },
)
```

`LlamaIndexLLMSelector` uses your native LlamaIndex LLM unchanged. Subclass
`BaseActionSelector` and implement `async select(request)` to support another model family;
return `ActionSelection` containing `ToolCall` actions, or `None` to abstain.
Selectors implement the explicit abstract `ActionSelector` contract:
`requires_action_llm` controls loading an additional action-role LLM profile,
`shared_llm` provides an LLM to reuse for planning/helpers, and
`abstention_reason` supplies the message used when selection returns `None`.
`requires_action_candidates`, `screenshot_max_side`, and `llms` describe candidate,
vision-sizing, and telemetry requirements.
Built-ins inherit `BaseActionSelector`, which defaults to no additional action
LLM, no shared helper LLM, and the standard abstention message. Custom selectors
inherit this base for defaults or implement all abstract properties and `select`
directly on `ActionSelector`. Capabilities are accessed directly, not discovered
with `getattr` fallbacks.
Selectors never execute tools. `FallbackSelector` handles abstention, empty
selections, and configured exceptions before execution, not tool failures.
Add selectors to `fallbacks` in priority order; the last selector's exceptions
propagate if the chain is exhausted.

For an explicit, model-chosen handoff instead of an automatic fallback:

```python
from mobilerun.agent.selectors import DelegatingSelector

agent = MobileAgent(
    "Find Alice and send a message",
    action_selectors=DelegatingSelector(
        primary=DecisionSelector(jev),
        delegate=LlamaIndexLLMSelector(llm),
        description="Ask the LLM to choose the next action when text entry or another unsupported action is needed.",
    ),
)
```

The primary can choose a virtual `delegate` action. The second selector receives
the original observation and selects this step's action; the next step starts
with the primary again. Reverse the selectors for LLM → Jev delegation. The
virtual action is never executed by the tool registry and must be selected alone.
Errors and abstention propagate normally; wrap the chain in `FallbackSelector`
if automatic recovery is also wanted. Supply `config` when planning/helper LLMs
are needed; the delegate LLM is not automatically reused for those roles.

The CLI also supports delegation through `agent.delegation` in YAML:

```yaml
agent:
  delegation:
    primary: fast_agent
    delegate: delegate
    description: Ask the LLM to generate and type text when needed.
```

Each name refers to a profile in `decision_profiles` or `llm_profiles`;
decision profiles take precedence if both have the same name. Omit `primary`
to use the active `fast_agent`/`executor` role. Omit `delegation` (or set it to
`null`) to keep the normal single-selector behavior. Explicit selectors,
including CLI `--provider`/`--model` overrides, take precedence over this config.

`jev-test-delegation.yaml` configures Jev → OpenAI delegation:

```bash
uv run --extra typesafe --env-file .env mobilerun run \
  "Check whether a timer is running in Clock. If none is running, write a six-sentence note reminding me to set a timer tomorrow." \
  --config jev-test-delegation.yaml --device emulator-5554 --steps 30
```

Decision models implement `async decide(request)` and return
`Decision(option_index=...)`, using a zero-based index into `request.options`.
Missing/empty options and invalid indices raise `ValueError`. Providers supply
deterministic click/long-press candidates; unavailable tools are filtered out.
Override `StateProvider.get_action_candidates(state)` for other actions or
parameterized input. The request includes the goal, UI, available tools,
variables, screenshot, recent history, and persistent memory separately.

For decision-only FastAgent completion, return
`Decision(success=True, message="Finished")` (or `success=False`); this uses
the existing `complete` tool and is rejected by ExecutorAgent. A selector-only
MobileAgent can run without config or an LLM. Named Android app opening and
structured-output extraction require their helper LLMs. When config is supplied,
the original LLM-profile loading and dictionary-merging behavior is preserved.

Decision providers can be loaded separately from the workflow:

```python
from mobilerun.agent.utils.decision_loader import (
    load_decision_model,
    register_decision_provider,
)

# MyDecisionModel implements async decide(request).
register_decision_provider("my-provider", MyDecisionModel)
decision_model = load_decision_model("my-provider", model="my-model")
```

Provider factories receive constructor options unchanged. No decision-provider
SDK or transport is assumed; register your integration's factory explicitly.

#### Decision-model prompts

Native decision models use `mobilerun/config/prompts/decision/system.jinja2`,
rendered with the same `PromptLoader` and `PromptResolver` as LLM agents.
The default prompt describes only selectable action candidates: without a typing
candidate it avoids search fields, and with swipe candidates it explains scrolling.

Configure a custom template file:

```yaml
agent:
  decision:
    system_prompt: prompts/my_decision_system.jinja2
```

Or supply an inline Jinja2 template with
`MobileAgent(..., prompts={"decision_system": "..."})`; inline overrides take
precedence over the file. Templates receive `goal`, `protocol`, `available_tools`
(candidate tools plus completion when supported), `registered_tools`, `can_type`,
`can_scroll`, `can_complete`, `vision`, and `variables`. The rendered instructions
are passed to decision models as `SelectionRequest.instructions`, including in
fallback chains. LLM selectors continue using their existing chat prompts.

## ⚙️ Features

- **CLI:** Run one-off natural language tasks, inspect devices, replay macros, and debug from the terminal.
- **Python API:** Build custom mobile automation workflows with Python and use custom tools.
- **Android and iOS support:** Control Android through the Portal app, iPhones through `mobilerun-ios`, or the iOS Simulator through the iOS Portal app.
- **Portal-based control:** Use UI trees, screenshots, text input, gestures, app launching, and device state from the Portal runtime.
- **Vision mode:** Send screenshots to the LLM with `--vision`, or use screenshot-only control with `--vision-only` (useful for the apps that do not have a11y tree information).
- **Reasoning mode:** Use `--reasoning` for manager-executor planning on longer or more complex tasks.
- **Tracing and telemetry:** Debug execution with Arize Phoenix, Langfuse, saved trajectories, and detailed logs.
- **Structured output:** Return structured data from mobile workflows.
- **App cards and custom tools:** Add app-specific guidance to make agent perform better on your use-cases.
- **Docker:** Run Mobilerun in a container for repeatable local environments.

## ☁️ Framework vs Cloud

| | Mobilerun Framework | Mobilerun Cloud |
| --- | --- | --- |
| Best for | Running agents locally on your own machine and devices | Ready-to-go local phone control, hosted real or virtual devices, API workflows, and managed device operations |
| Runtime | Your machine  | Mobilerun-managed infrastructure |
| Interface | CLI, Docker, and Python API | Dashboard, REST API, SDKs, and hosted devices |

Use the framework when you want full local control of the agent runtime. Use [Mobilerun Cloud](https://cloud.mobilerun.ai) when you want managed devices, fleet workflows, or cloud APIs without running the agent locally. Learn more in the [framework overview](https://docs.mobilerun.ai/framework/overview) and the [cloud docs](https://docs.mobilerun.ai).

### Which should I choose?

- Choose **Mobilerun Framework** for local agent execution and code-level control.
- Choose **Mobilerun Cloud** for managed phones, APIs, and scale without running agents locally.

### Cloud Device Types

| Device type | What it is | Best for |
| --- | --- | --- |
| Personal | Your own hardware connected to Mobilerun Cloud | Quick automation on devices you own |
| Cloud Phone (Hosted) | Instantly available cloud-hosted phone | Scalable hosted automation |
| Physical Phone (Hosted) | Real hardware, not emulated | QA and workflows that need real device behavior (sensors, camera, push, payments) |

### Run an agent on a Cloud Phone

Ask an agent to test the onboarding flow in your app: open the app, complete the welcome screens, sign in with a test account, and confirm that the home screen loads. Watch each step live and inspect the results to diagnose failures.

### Generate leads with Google Maps on a Cloud Phone

<a href="https://www.youtube.com/watch?v=CaL71WorysA">
  <img src="./static/google-maps.gif" alt="Mobilerun generating business leads with Google Maps on a cloud device" width="800">
</a>

## 🧩 Agent Skill

Give Claude Code, Codex, or another coding agent direct control of Android and iOS devices with the [mobile-harness](https://github.com/droidrun/mobile-harness) skill. Copy paste it into your agent:

```text
Set up https://github.com/droidrun/mobile-harness for me.

Read `install.md` and follow the steps to install `mobile-harness`.
```

The skill teaches the agent to drive local devices (ADB, Portal, iPhones, iOS Simulator) and Mobilerun Cloud phones.

## 🎬 Demo Videos

### Book accommodation from a prompt

Shows multi-step navigation, text input, and app-state reasoning while Mobilerun searches for accommodation.

<a href="https://youtu.be/VUpCyq1PSXw">
  <img src="./static/demo-apartment-search.gif" alt="Mobilerun booking accommodation from a prompt" width="800">
</a>

### Find trending content

Shows browsing, app navigation, and result extraction from a natural-language task.

<a href="https://youtu.be/7V8S2f8PnkQ">
  <img src="./static/demo-reddit-trends.gif" alt="Mobilerun finding trending content from a prompt" width="800">
</a>

## 💡 Example Use Cases

- Mobile app QA and regression testing
- Guided workflows for non-technical users
- Repetitive task automation on mobile devices
- Event-driven automation from schedules, notifications, or custom triggers
- Data extraction from native mobile apps
- Testing the same flow across many devices and OS versions in parallel

## 🌐 Part of Mobilerun

Mobilerun is one piece of a toolkit for driving real phones with AI agents:

- **[Mobilerun Portal](https://github.com/droidrun/mobilerun-portal)**: the Android app that lets you control and automate your phone.
- **[mobilerun-ios](https://github.com/droidrun/mobilerun-ios-releases)**: connects an iPhone over USB, either to this framework (`--local`) or to Mobilerun Cloud.
- **[iOS Portal](https://github.com/droidrun/ios-portal)**: the iOS app that lets the framework drive the iOS Simulator.
- **[mobile-harness](https://github.com/droidrun/mobile-harness)**: an agent skill that lets Claude Code, Codex, and OpenClaw control Android, iOS, and cloud phones. App cards teach the agent how each app works, and shared memory keeps what it learns across tasks. It scores 100% on AndroidWorld and 98.34% on iOSWorld ([benchmark results](https://mobilerun.ai/benchmark/)).
- **[Mobilerun Cloud](https://mobilerun.ai)**: cloud phones for AI agents. Hosted Android phones and dedicated iPhones for testing and running your agents on real devices, with phone numbers and inboxes so end to end flows like sign in and verification work in your own apps, devices in multiple regions for localized testing, and scheduled and event triggers. Drive it through the REST API, [Python](https://github.com/droidrun/mobilerun-sdk-python) and [TypeScript](https://github.com/droidrun/mobilerun-sdk-typescript) SDKs, or MCP, and watch every step live.

## 📚 Documentation

- [Framework quickstart](https://docs.mobilerun.ai/framework/quickstart)
- [Mobilerun cloud quickstart](https://docs.mobilerun.ai/quickstart)
- [Device setup](https://docs.mobilerun.ai/framework/guides/device-setup)
- [CLI guide](https://docs.mobilerun.ai/framework/guides/cli)
- [SDK reference](https://docs.mobilerun.ai/framework/sdk/reference)
- [Custom tools](https://docs.mobilerun.ai/framework/features/custom-tools)
- [Agent architecture](https://docs.mobilerun.ai/framework/concepts/architecture)
- [Structured output](https://docs.mobilerun.ai/framework/features/structured-output)
- [Tracing](https://docs.mobilerun.ai/framework/features/tracing)

## 👥 Contributing

Contributions are welcome. Please feel free to submit a pull request or open an issue.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

## Security Checks

To help catch security issues before submitting changes, run:

```bash
bandit -r mobilerun
safety scan
```
