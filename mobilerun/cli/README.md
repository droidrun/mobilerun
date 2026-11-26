# cli/

Core's CLI. Feature packages keep their own CLI inside their folder.

Command-line entry point and command groups. Run `mobilerun COMMAND --help` for flags and arguments. Old top-level aliases are unavailable; use `account login`, `device list`, or `agent run` instead.

## Quick example

```sh
mobilerun device list
mobilerun device input tap 200 200 emulator-5554
```

## Command tree

```text
mobilerun
├── account: login, logout, status
├── agent: llm (configure, login), replay, run, task (status, stop, trajectory)
├── completion
├── credentials: delete, list, path, save, update
├── device
│   ├── list, connect, disconnect, ping, setup
│   ├── action: call, extensions, list
│   ├── app: install, list, open, start, stop, uninstall
│   ├── cloud: reboot, resume, status, stop, terminate
│   ├── file: list, pull, push, rm
│   ├── input: clipboard (get, set), long-press, press, swipe, tap, type
│   ├── screen: screenshot, ui
│   └── settings: language (get, set), location (get, reset, set), timezone (get, set)
├── doctor
├── macro: list, replay
├── mailbox: code, create, included, list, release, status
├── mcp: install, uninstall, list
├── number: bind, code, countries, create, included, list, messages, release, send, status, unbind
├── proxy: bind, buy, included, list, release, status
├── skills: install, list, show, uninstall
├── webhook: delete, get, list, ping, register
└── workspace: apps (list, remove, upload), files (download, list, rm, upload)
```

`agent llm login` also offers provider subcommands: `anthropic`, `gemini`, `openai`, and `xai`.

Cloud resources and agent runs can incur charges. Device commands select an explicit
device first, then a saved default or a discovered device; see [device commands](devices/README.md).
Overwriting an existing credential requires an interactive confirmation.

`number create`, `mailbox create`, `proxy buy`, `number send`, and `device cloud
resume` ask only when a billing guard rejects the chosen mode. The prompt defaults
to No; Yes retries the guarded operation once with the mode named in the prompt.
No cancels with exit status 1. `--yes` approves that retry without a prompt;
selecting `paid` authorizes charges up front. Destructive actions still require
separate confirmation.

## Contents

| Path | Purpose |
| --- | --- |
| `app.py` | Root command and lazy registration |
| [auth/](auth/README.md), [credentials/](credentials/README.md) | Authentication and credentials |
| [base/](base/README.md), `doctor.py` | Shared CLI helpers and diagnostics |
| [devices/](devices/README.md) | Device commands and selection |
| [mailbox/](mailbox/README.md), [phone_number/](phone_number/README.md), [proxy/](proxy/README.md) | Cloud resources |
| [workspace/](workspace/README.md) | Workspace files and apps |
| [webhook/](webhook/README.md) | Cloud webhook endpoints and one-time signing secrets |
| [macro/](../macro/cli/README.md), [skills/](../skills/cli/README.md) | Macro and skill commands |
