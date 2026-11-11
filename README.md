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

Use Python 3.11–3.13. Python 3.14 is not supported.
Install the CLI with [`uv`](https://docs.astral.sh/uv/):

```bash
uv tool install --python 3.13 mobilerun
```

For Anthropic models, include the optional extra:

```bash
uv tool install --force --python 3.13 "mobilerun[anthropic]"
```

For Python integration, install into a virtual environment instead:

```bash
uv venv --python 3.13
uv pip install mobilerun
```

## 🚀 Quickstart: local Android

Install [Android SDK Platform-Tools](https://developer.android.com/studio/releases/platform-tools)
so `adb` is on your PATH. Connect one device using either QR pairing or USB.

**QR pairing:** on Android 11 or later, put the device and computer on the same Wi-Fi.
Open Developer options → Wireless debugging → "Pair device with QR code", then scan:

```bash
mobilerun device pair
```

Keep this command in a terminal. It prints the ready device serial after adb
connects. The timeout is 120 seconds; `--timeout SECONDS` changes it.

**USB:** enable Developer options and USB debugging, connect the device, and approve
its debugging prompt. Check the connection:

```bash
mobilerun device list
```

Install the Portal, check device access, configure an LLM provider, and run a task:

```bash
mobilerun device setup
mobilerun device ping
mobilerun configure
mobilerun run "Open Settings and report the Android version"
```

The setup command installs the Portal and enables its accessibility service.
The configure wizard selects a provider, authentication method, and model.
If more than one device is connected, pass `-d DEVICE_SERIAL` to each command
that selects a device. Replace uppercase placeholders with your own values.

For iOS, follow the [device setup guide](https://docs.mobilerun.ai/framework/guides/device-setup)
for `mobilerun-ios` over USB or the iOS Portal for the simulator.

### Quickstart: cloud

Use an existing device in [Mobilerun Cloud](https://cloud.mobilerun.ai).
Sign in and copy its id from the listing:

```bash
mobilerun auth login
mobilerun device list -t cloud
```

Sign-in prints a link, a user code, and a QR code when stderr is a terminal.
Scan the QR or use the link and code to approve access. This cloud sign-in is
separate from the LLM provider configured above.

Set the target and default device, replacing the placeholder with the listed id:

```bash
export MOBILERUN_TARGET=cloud
export MOBILERUN_DEVICE="CLOUD_DEVICE_ID"
mobilerun run "Open Settings and report the Android version" -d "$MOBILERUN_DEVICE"
```

The environment values also apply to later commands, so you can omit the device
flag after setting the default. An explicit device selects its backend: a cloud
id, an ADB serial, or a Portal HTTP URL. Without a device, the local target
selects an attached device; the cloud target requires an id.

### Command tree

```text
mobilerun
├── run TASK
├── device
│   ├── list
│   ├── pair
│   ├── connect HOST:PORT
│   ├── disconnect HOST:PORT
│   ├── setup
│   ├── ping
│   ├── tap X Y
│   ├── swipe X1 Y1 X2 Y2
│   ├── long-press X Y
│   ├── type TEXT
│   ├── press BUTTON
│   ├── screenshot
│   ├── ui
│   ├── apps
│   └── start PACKAGE
├── configure
├── auth
│   ├── login [SERVICE]
│   ├── logout [SERVICE]
│   └── status
├── doctor
├── env
├── macro
│   ├── replay PATH
│   └── list [DIRECTORY]
└── completion SHELL
```

The sign-in services are `mobilerun` (default), `openai`, `anthropic`, `gemini`,
and `xai`. Provider sign-ins use the same auth group; the configure command
runs the wizard. The `configure` provider choices include only supported
providers; `run` and `macro replay` also accept `openrouter` and `deepseek`.
Help includes examples, and command typos suggest a correction without executing
it. The `run --agent` option appears only when an external agent is installed.
See each command's help for its arguments and options.

### Shared flags

A short letter has one meaning. Options apply only to commands that list them
in help; local pairing and TCP connect/disconnect do not accept device-selection
flags. In the table, **device actions** means tap, swipe, long-press, type, press,
screenshot, ui, apps, start, and ping.

| Short | Long | Commands / meaning |
| --- | --- | --- |
| `-h` | `--help` | Every command: show usage. |
| `-V` | `--version` | Root: show the version. |
| `-v` | `--verbose` | Every command: enable logs; off unless enabled in config or environment. |
| `-d` | `--device` | Device actions, setup, run, doctor, macro replay: device serial, Portal URL, or cloud id. |
| `-t` | `--target` | Device actions, list, run, doctor, macro replay: `local` (default) or `cloud`. |
| `-P` | `--platform` | Device actions, run, doctor, macro replay: `android` or `ios`; otherwise use config. |
| `-c` | `--config` | Device actions, setup, run, doctor, configure, env, macro replay: config file. |
| `-j` | `--json` | Device list/ui/apps/screenshot, auth status, doctor, env, macro list, run: one JSON document. |
| `-p` | `--provider` | Run, configure, macro replay: LLM provider. |
| `-m` | `--model` | Run, configure, macro replay, supported provider logins: LLM model. |
| `-u` | `--llm-url` | Run and configure: LLM endpoint, when supported by the provider. |
| `-s` | `--max-steps` | Run and macro replay: step limit. |
| `-T` | `--temperature` | Run: sampling temperature. |
| `-a` | `--agent` | Run: installed external agent. |
| `-r` | `--reasoning` / `--no-reasoning` | Run: enable or disable reasoning. |
| `-n` | `--dry-run` | Macro replay: show actions without executing them. |
| `-k` | `--api-key` | Configure: key for the selected LLM provider. |
| — | `--api-url` | Device actions, list, run, doctor, macro replay: cloud API endpoint. |
| — | `--auth-url` | Mobilerun login: sign-in endpoint. |
| — | `--tcp` / `--no-tcp` | Run and macro replay: local Android TCP mode. |
| — | `--no-qr` | Mobilerun login: suppress the QR code. |
| — | `--no-input` | Configure and auth login: never prompt for input. |

### Environment

For CLI options, flags override environment values, then config values and defaults.
Inspect resolved values and sources with `mobilerun env`; it shows secrets only
as `set` or `not set`. Provider authentication follows the selected provider's
configuration and credential store. `auth status` always checks the service
that issued the saved session, even when `MOBILERUN_AUTH_URL` has changed.

| Variable | Meaning | Default / fallback |
| --- | --- | --- |
| `MOBILERUN_TARGET` | Default target | `local` |
| `MOBILERUN_DEVICE` | Default device serial, Portal URL, or cloud id | Configured serial, then local selection |
| `MOBILERUN_CLOUD_API_KEY` | Cloud authentication | Saved Mobilerun sign-in credential |
| `MOBILERUN_API_BASE_URL` | Cloud API endpoint | `https://api.mobilerun.ai/v1` |
| `MOBILERUN_AUTH_URL` | Mobilerun login endpoint | `https://cloud.mobilerun.ai/api/auth` |
| `MOBILERUN_CONFIG` | Config file | Platform user config directory |
| `MOBILERUN_ANDROID_PORTAL_TOKEN` | Android Portal HTTP token | Config token only when its platform is Android |
| `MOBILERUN_IOS_PORTAL_TOKEN` | iOS Portal HTTP token | Config token only when its platform is iOS |
| `MOBILERUN_DEBUG` | Verbose logs | Config, otherwise off |
| `MOBILERUN_TELEMETRY_ENABLED` | Anonymous telemetry | Enabled unless config or environment disables it |
| `MOBILERUN_STREAM_SCREENSHOTS` | Stream agent screenshots | Off |
| `MOBILERUN_OAUTH_MANUAL` | Use manual provider sign-in | Off; headless sessions may select a device flow |
| `ANDROID_SERIAL` | Local ADB selection fallback | No value |
| `NO_COLOR` | Disable colour; bold may remain | No value |
| `PHOENIX_URL` | Phoenix trace endpoint | `http://0.0.0.0:6006` |
| `PHOENIX_PROJECT_NAME` | Phoenix project | Empty |
| `GOOGLE_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `MINIMAX_API_KEY`, `ZAI_API_KEY` | LLM provider credentials | Selected provider configuration / saved credentials |
| `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY` | Credentials for run-only providers | Provider defaults |

### Scripting

Stdout contains data. Progress and error messages go to stderr.
`device list`, `device apps`, and `macro list` print tab-separated rows;
a header row appears only when stdout is a terminal. Use `-j` for the stable
machine-readable form. `mobilerun env` prints `NAME=value  (source)`, with empty
unset values, lowercase booleans, and secrets shown only as `set` or `not set`.
The text UI output has one tab-separated row per element: index, class, text,
resource id, bounds, and flags. Filter it with standard tools:

```bash
mobilerun device ui | grep -i 'settings'
mobilerun device ui -j | jq '.a11y_tree'
mobilerun device list -t cloud -j | jq '.[].device_id'
```

Install `jq` to use the JSON examples. UI text uses the filtered tree;
`device ui --all` also includes empty layout containers. That option cannot
be combined with JSON output.

Exit codes are **0** for success, **1** for failure, **2** for invalid usage,
and **130** for interruption. Read commands with JSON output return an error
document on failure; API errors include the HTTP status and an optional code:

```json
{"error":"Device not found","status":404,"code":null}
```

`doctor -j` retains completed checks, including when it exits 1:

```json
{"ok":false,"checks":[{"name":"Device","status":"fail","detail":"no device connected\nConnect a device over USB or run `mobilerun device pair`; use `-t cloud` for cloud devices."}]}
```

`device list --all` lists local devices first, then cloud devices. If either
listing fails, the other result is kept and a warning goes to stderr. It exits
1 only when both listings fail; a successful empty listing still counts as
success. With `-j`, it returns `{"devices":[...],"errors":[{"target":"cloud","error":"..."}]}`.
Listing a single target keeps the plain JSON list on success and fails normally
when that target is unavailable.

Other errors contain only `error`. Usage errors and interruptions may leave
stdout empty. The run command emits `success`, `reason`, and `steps` after a
task, including a failed task; errors before execution use the error document.
Check the command's exit status as well as its JSON.

### Non-interactive use and secrets

`configure` and `auth login` never prompt when stdin is not a terminal or
`--no-input` is supplied. Missing configuration options fail with exit 2 and
name the option to provide. For example, supply `-p`, `-m`, and a provider key
through its environment variable or stdin. An OAuth configuration needs an
existing sign-in when input is disabled. Prompt EOF exits 1; Ctrl-C exits 130.

Use `-k -` or `--token -` to read one secret line from stdin. The trailing newline
is stripped, and empty input fails with exit 2. A literal flag value still works,
but it remains in shell history; prefer stdin or the provider's environment
variable. These examples use shell variables that are already set:

```bash
printf '%s\n' "$OPENAI_API_KEY" | mobilerun configure -p openai -m MODEL -k - --no-input
printf '%s\n' "$ANTHROPIC_SETUP_TOKEN" | mobilerun auth login anthropic --token - --no-input
```

Device-code and browser-callback login flows can proceed without terminal input.
If a provider flow needs a pasted code, it fails before starting when input is
disabled. For Anthropic, supply `--token -`; for Gemini, run the login in a
terminal without `--no-input`: headless Gemini sign-in needs a terminal because
no option supplies its code. Tokens and keys are never included in CLI errors
or verbose logs.

### Shell completion

Print a completion script with `mobilerun completion bash|zsh|fish`. The command
writes only the script; it does not install anything. Add the matching line to
your shell configuration (Bash 4.4 or later is required for Bash completion):

```bash
# ~/.bashrc
eval "$(mobilerun completion bash)"
```

```zsh
# ~/.zshrc (after compinit)
eval "$(mobilerun completion zsh)"
```

```fish
# ~/.config/fish/config.fish
mobilerun completion fish | source
```

### Upgrading from 0.7

The old spellings are removed; there are no aliases. Replace them as follows:

| Before | After |
| --- | --- |
| `mobilerun devices` | `mobilerun device list` |
| `mobilerun connect SERIAL`, `mobilerun disconnect SERIAL` | `mobilerun device connect SERIAL`, `mobilerun device disconnect SERIAL` |
| `mobilerun setup`, `mobilerun ping` | `mobilerun device setup`, `mobilerun device ping` |
| `mobilerun login`, `mobilerun logout`, `mobilerun whoami` | `mobilerun auth login`, `mobilerun auth logout`, `mobilerun auth status` |
| `mobilerun configure openai`, `mobilerun configure anthropic`, `mobilerun configure gemini`, `mobilerun configure xai` | `mobilerun auth login SERVICE`; use `mobilerun configure -p PROVIDER` for the wizard |
| Device selection `--device-id` | `-d` / `--device` |
| Cloud `--base-url` | `--api-url` |
| `--cloud`, `--ios` | `-t cloud`, `-P ios` |
| `--debug` / `--no-debug` | `-v` / `--verbose`; omit to use config/default |
| Run `--base_url`, `--api_base` | `-u` / `--llm-url` |
| Run `--steps` | `-s` / `--max-steps` |
| Run `--control-backend`, `--device-id` | `--backend`, `--backend-device` for visual remote control |
| Run `--no-vision-only` | Omit `--vision-only` |
| Setup `--path`, `-pv` | `--apk`, `--portal-version` |
| Macro replay `-m` (step limit), `-t` (delay), `-s` (start step) | `-s` / `--max-steps`, `--delay`, `--start-from`; `-m` now selects a model |
| `DROIDRUN_CONFIG`, `DROIDRUN_TELEMETRY_ENABLED` | `MOBILERUN_CONFIG`, `MOBILERUN_TELEMETRY_ENABLED` |
| `DROIDRUN_STREAM_SCREENSHOTS`, `DROIDRUN_OAUTH_MANUAL` | `MOBILERUN_STREAM_SCREENSHOTS`, `MOBILERUN_OAUTH_MANUAL` |
| `MOBILERUN_DEVICE_TOKEN` | Platform-specific Portal token variables in the environment table |
| `phoenix_url`, `phoenix_project_name` | `PHOENIX_URL`, `PHOENIX_PROJECT_NAME` |

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

`type_secret` types a value from your `CredentialManager`. Typed secrets are masked in UI trees, logs and saved macros: in every UI tree the agent sees (LLM prompts, trajectories, saved macros) and in the engine's log output. Screenshots keep showing ordinary typed secrets; only screenshots after approval-gated secrets are black.

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

- The value is resolved only after approval, and typed only if the last observed state still shows the approved app (and, in a browser, the approved host), with the field you selected focused: the focused node must be that very element (resource id, class, tree position and bounds) and a text field, not an enclosing container or another node with the same bounds. The device offers no atomic "check and type", so this is a check of the last observed state; a focus change in the instant before typing cannot be excluded.
- Approval-gated secrets need a structured accessibility tree (Android, iOS portal-http). On the iOS XCTest text tree they fail closed: nothing is typed.
- Browser hosts are read only from the address bar of known browsers (Chrome, Samsung Internet, Firefox, Edge, Brave and DuckDuckGo on Android (the DuckDuckGo adapter and its address-bar id are not verified on a device); Safari on iOS only where a structured tree with a `URL` field exists in the path), and only from a text node outside web content. An unreadable or ambiguous host, or a visible address bar with a well-known id in a browser that is not on this list, means nothing is typed. A browser that is not on the list and shows no such address bar is bound to its package only, like a WebView inside an ordinary app. A host can deny approval requests that lack `targetHost` for browser packages.
- From the first typed approval-gated secret on, screenshots from the engine's driver are black for the rest of the run. Typed ordinary secrets (passwords, tokens) are masked in UI trees only.
- A macro never replays an approval-gated step; replay stops with `approval_required`.

Limits to know about:

- `redact="input_only"` masks a value only in focused or editable nodes, where the tree says which are which (Android). Without that information, including the iOS text tree, the value is masked everywhere. Digit-only values of at least 8 digits are always masked everywhere. Short values may remain visible in nodes that are not editable.
- Whole-word matching cannot catch a value that an app glues to other characters. A value of at least four characters that an app spreads over adjacent text nodes (a code with one digit per node, a card number in group nodes) is masked in every contributing node, however large the screen is; each such node may only use characters of the value, so a PIN pad with other keys or letters is left alone. For `input_only` secrets only input nodes are joined. For a host-injected state provider, the pre-formatted text of its state is masked per string only.
- If typing fails or is interrupted, every typed prefix of at least four characters is masked, in input nodes (prefixes of eight or more digits everywhere). Shorter prefixes stay unmasked.
- The destination check refuses when the tap restructures the screen's ancestors (the focused node must keep the same tree position); that is fail closed.
- A replay with `on_mismatch="agent"` that typed any secret returns `False` instead of handing off to the agent.
- Target binding covers the typing step only; copy/paste by the app or the user can move a value elsewhere.
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
