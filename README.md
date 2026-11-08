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

## 🔐 Approval for sensitive secrets

`type_secret` types a value from your `CredentialManager` without showing it to the model. Typed values are masked in every UI tree the agent sees (LLM prompts, trajectories, saved macros) and in the engine's log output.

For secrets that need a human decision first (a payment card, for example), describe them with `SecretSpec` and pass an `ApprovalProvider` to the agent. The engine asks once per run, per secret group and target, and types only after the answer is `"approved"`. A `"denied"` or `"expired"` answer, a failing provider, or no provider at all ends the run as failed (`approval_denied`, `approval_expired`, `approval_unavailable`) and nothing is typed.

```python
from mobilerun import MobileAgent
from mobilerun.credential_manager import (
    ApprovalDecision,
    ApprovalProvider,
    CredentialManager,
    SecretSpec,
)


class TerminalApproval(ApprovalProvider):
    async def request(self, kind: str, details: dict[str, str]) -> ApprovalDecision:
        target = details.get("targetHost") or details["targetPackage"]
        answer = input(f"Allow '{details['label']}' ({kind}) to be typed into {target}? [y/N] ")
        return "approved" if answer.strip().lower() == "y" else "denied"


class MyCredentials(CredentialManager):
    async def resolve_key(self, key: str) -> str:
        return load_from_your_vault(key)

    async def get_keys(self) -> list[str]:
        return ["card_number", "card_cvc", "login_password"]

    async def get_secret_specs(self) -> list[SecretSpec]:
        card = dict(requires_approval=True, approval_kind="card", approval_label="Test card")
        return [
            SecretSpec("card_number", **card),
            SecretSpec("card_cvc", **card, redact="input_only"),
            SecretSpec("login_password"),
        ]


agent = MobileAgent(
    "Pay the open invoice",
    config=config,
    credentials=MyCredentials(),
    approval_provider=TerminalApproval(),
)
```

`request()` receives `label`, `secretGroup` and `targetPackage`, plus `targetHost` in a browser. These are never secret values. One approval covers every field of the same kind, label and target, and nothing else. Who is asked, how the decision is authenticated and how it travels back are up to your provider.

What the engine guarantees for approval-gated secrets:

- The value is resolved only after approval, and typed only if the last observed state still shows the approved app (and, in a browser, the approved host), with the field you selected focused and editable. The device offers no atomic "check and type", so this is a check of the last observed state; a focus change in the instant before typing cannot be excluded.
- Browser hosts are read only from the address bar of known browsers (Chrome, Samsung Internet, Firefox, Edge, Brave on Android, Safari on iOS). An unreadable or ambiguous host, or an address bar of an unknown browser, means nothing is typed. In WebViews inside ordinary apps the approval is bound to the app only.
- From the first typed approval-gated secret on, screenshots from the engine's driver are black for the rest of the run. Typed ordinary secrets (passwords, tokens) are masked in UI trees only.
- A macro never replays an approval-gated step; replay stops with `approval_required`.

Limits to know about:

- `redact="input_only"` masks a value only in focused or editable nodes, where the tree says which are which (Android). Without that information, including the iOS text tree, the value is masked everywhere. Digit-only values of at least 8 digits are always masked everywhere. Short values may remain visible in nodes that are not editable.
- Whole-word matching cannot catch a value that an app glues to other characters.
- A token shown as plain text in a screenshot after typing an ordinary secret stays visible; only approval-gated secrets turn screenshots black.
- Coverage is the built-in workflows and cooperating state providers (their `UIState` is masked at the engine boundary; the provider itself is not modified). A host that reads the raw driver itself, a live stream or screen recording of the device, host-side log handlers, telemetry exporters, and external agents with raw ADB access are outside this boundary.
- Screenshot-only mode (`--vision-only` without accessibility data) has no `type_secret`, so approval-gated secrets are unavailable there. Hybrid vision with accessibility data is supported.
- Apps whose card fields appear neither in the accessibility tree nor as focus cannot be filled; the field check fails and nothing is typed.
- The black screenshots take the agent's sight for the rest of a vision run; after an approval-gated secret it should only submit.

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
