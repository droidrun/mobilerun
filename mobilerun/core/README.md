# core/

**Could someone delete this folder and still use the rest of Mobilerun?**
If yes, it belongs in an optional feature (`sandbox/`,
`mcp/`, `macro/`, `skills/`); if not, put it in `core/` or a shared package.
For example, `sandbox/` is removable; `core/webhooks/` is not, because the
webhook CLI and sandbox both use it. In `mobilerun/`, `core/` holds domain
objects: devices, operators, state, credentials, proxies, numbers, mailboxes,
webhooks, workspace, and extensions. `models/`, `clients/`, `errors/`, `utils/`,
`config/`, and `logs/` supply shared building blocks. Phone behavior lives in
`core/devices/` and its components; account-wide resources live in their own
core folders or on `Workspace`.

## Topic map

Apps and files each have two entry points: `Workspace` manages the account
library, while `Device` methods act on one phone. For example,
`Workspace.delete_app()` removes a library entry; `device.uninstall_app()` does not.
`—` means there is no dedicated shared model or cloud adapter for that topic.

| Topic | User-facing API | Domain code | Model | Cloud API |
| --- | --- | --- | --- | --- |
| Devices | `Device`, `CloudDevice`, `AndroidDevice`, `IOSDevice`, `RemoteDevice`, `DevicePool` | [devices/](devices/) | [Plan](../models/billing.py) | [devices.py](../clients/cloud/devices.py) |
| Apps | `Workspace.upload_app()`, `Workspace.list_apps()`, `device.install_app()`, `device.uninstall_app()` | [workspace.py](workspace.py), [cloud_apps.py](devices/components/cloud_apps.py), [base.py](devices/base.py) | [App](../models/apps.py) | [apps.py](../clients/cloud/apps.py) |
| Files | `Workspace.upload_file()`, `Workspace.list_files()`, `device.push_file()`, `device.pull_file()` | [workspace.py](workspace.py), [device files](devices/components/files.py) | [WorkspaceFile, DeviceFile](../models/files.py) | [files.py](../clients/cloud/files.py) |
| Settings/environment | `device.get_location()`, `device.set_location()`, language and timezone methods | [environment.py](devices/components/environment.py) | [Location](../models/location.py) | [devices.py](../clients/cloud/devices.py) |
| Proxies | `CloudRotatingProxy`, `CloudFixedProxy`, `CloudSocks5Proxy`, `LocalProxy`; `proxy.bind(device)` | [proxy/](proxy/) | [ProxyStatus](../models/proxy.py) | [proxies.py](../clients/cloud/proxies.py) |
| Numbers | `CloudPhoneNumber`, `device.number` | [phone_number/](phone_number/) | — | [numbers.py](../clients/cloud/numbers.py) |
| Mailboxes | `CloudMailbox`, `device.mailbox` | [mailbox/](mailbox/) | — | [mailboxes.py](../clients/cloud/mailboxes.py) |
| Credentials | `CloudCredentials`, `EnvCredentials`, `FileCredentials`, `DictCredentials` | [credentials/](credentials/) | [Credential](../models/credentials.py) | [secrets.py](../clients/cloud/secrets.py) |
| Webhooks | `CloudWebhooks` | [webhooks/](webhooks/) | [WebhookEndpoint](../models/webhooks.py) | [webhooks.py](../clients/cloud/webhooks.py) |
| Operators/tasks | `device.do()`, `device.expect()`, `Operator` | [operator/](operator/) | [ExecutionResult, ExpectationResult](../models/results.py), [CloudTaskInputs](../models/cloud_task.py) | [tasks.py](../clients/cloud/tasks.py) |
| State | `device.state`, `StateProvider`, `UIState` | [state/](state/) | [UIElement](../models/ui.py) | — |
| Recording | `RecordingDevice`, `recorded_actions` | [recording.py](devices/recording.py) | [RecordedAction](../models/recording.py) | — |
| Extensions | `Extension.bind()`, `device.extensions`, `device.actions` | [extensions/](extensions/) | [ContextItem](../models/context.py) | — |

**Examples:**

Use an existing `device`.

```python
state = await device.state.refresh()
print(state.formatted_text)
```

**Depends on / used by:** Uses clients and other shared packages; used by devices and features.

See [devices](devices/README.md), [state](state/README.md), [webhooks](webhooks/README.md), and [Workspace](WORKSPACE.md).
