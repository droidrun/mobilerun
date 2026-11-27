# Demos

Examples for exploring Mobilerun from a source checkout. Run commands from the repository root.

## Start here

From the repository root, with a local Android emulator and Portal running:

```bash
python -m tests.demos.macro_demo --serial emulator-5554
```

Record and replay a short sequence without AI or cloud charges. For a cloud vault credential on the same device, run `python -m tests.demos.vault_demo` (cloud account required). To create a cloud phone and run tasks, use `python -m tests.demos.task_demo`; **phone, proxy and tasks can cost money**.

## Demos

Run these from a source checkout with `python -m tests.demos.NAME`; demos are not included in the built wheel. Cloud demos require authentication and may change resources. Inspect their `--help` before running.

| Module | What it does | Cost |
| --- | --- | --- |
| [intro_demo](intro_demo.py) | Introduction to device APIs | Cloud and LLM use can cost money |
| [task_demo](task_demo.py) | Temporary cloud phone, tasks and typed output | Phone, proxy and tasks cost money |
| [app_demo](app_demo.py) | Installs an APK on local and cloud devices and runs tasks | Cloud and LLM use can cost money |
| [macro_demo](macro_demo.py) | Records and replays local Android actions | No cloud or LLM charges |
| [vault_demo](vault_demo.py) | Saves, uses and deletes a demo vault credential | Cloud account required; local device has no service cost |
| [phone_demo](phone_demo.py) | Rents a number and waits for an SMS code | Number and optional cloud phone can cost money |
| [mailbox_demo](mailbox_demo.py) | Creates a mailbox and waits for a code | Mailbox can cost money |
| [proxy_demo](proxy_demo.py) | Inspects or switches an existing phone's proxy | Proxy can cost money |
| [demo_billing](demo_billing.py) | Shared billing arguments and confirmation helper | Helper only; not a standalone demo |

`app_demo` has a hard-coded cloud device ID and APK setting; review them before use. Temporary resources are released by the demos unless retained with `--keep` or an existing resource is reused. Confirm cleanup in your cloud account after an interrupted run.

