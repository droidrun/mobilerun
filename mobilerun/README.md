# mobilerun/

Mobilerun connects device interfaces, execution operators, clients, and state providers. Its shared framework supports deterministic macros, sandbox apps, and coding-agent skills. LLM-backed execution and fallback require the separate `mobilerun-agents` add-on. It is not yet publicly published; there is no public pip or uv install command for it.

Pause with `await device.wait(seconds)`, not `asyncio.sleep()`: RecordingDevice
records device waits so macro replay keeps the pause.

See the [demos](../tests/demos/README.md) for source-checkout examples and costs.

## Package map

| Folder | Purpose |
| --- | --- |
| [addons/](addons/README.md) | Optional add-on discovery. |
| [core/](core/README.md) | Device, state, operator, and account-resource APIs. |
| [models/](models/) | Shared app, billing, credential and result value types; no IO or SDK. |
| [core/devices/](core/devices/README.md) | Device implementations. |
| [core/devices/base.py](core/devices/base.py) | Device protocol and shared mixin. |
| [core/devices/cloud.py](core/devices/cloud.py) | Cloud device, lifecycle, apps and task operator. |
| [core/devices/components/](core/devices/components/) | Driver, recording and stealth facades. |
| [core/phone_number/](core/phone_number/README.md) | Phone number interface and cloud implementation. |
| [core/mailbox/](core/mailbox/README.md) | Mailbox interface and cloud implementation. |
| [core/proxy/](core/proxy/README.md) | Proxy interface, Connect and SOCKS5 implementations. |
| [core/webhooks/](core/webhooks/README.md) | Organization webhook endpoint management. |
| [core/state/](core/state/README.md) | UI state providers and models. |
| [core/operator/](core/operator/README.md) | Operator interface. |
| [clients/](clients/README.md) | Shared service clients. |
| [clients/cloud/](clients/cloud/README.md) | Cloud account access. |
| [clients/portal/](clients/portal/README.md) | Portal client support. |
| [clients/local/](clients/local/README.md) | Functional local adapters and APK inspection. |
| [config/](config/README.md) | Shared configuration. |
| [core/credentials/](core/credentials/README.md) | Device-owned credential providers. |
| [utils/](utils/README.md) | Shared helpers. |
| [errors/](errors/README.md) | Shared errors. |
| [logs/](logs/README.md) | Shared logging. |
| [diagnostics/](diagnostics/README.md) | Diagnostics. |
| [assets/](assets/README.md) | Shared bundled assets. |
| [cli/](cli/README.md) | Root CLI and shared CLI helpers. |
| [cli/webhook/](cli/webhook/README.md) | Register, list, inspect, ping, and delete webhook endpoints. |
| [macro/](macro/README.md) | Recording and guarded replay. |
| [mcp/](mcp/README.md) | MCP configuration and installation. |
| [sandbox/](sandbox/README.md) | Sandbox apps and tasks. |
| [skills/](skills/README.md) | Bundled coding-agent skills. |
| `mobilerun-agents` (separate package) | Optional LLM agents, integrations and agent commands. |

## Where code goes

**Could someone delete this folder and still use the rest of Mobilerun?**
If yes, it belongs in an optional feature (`sandbox/`,
`mcp/`, `macro/`, `skills/`); if not, put it in `core/` or a shared package.
For example, `sandbox/` is removable; `core/webhooks/` is not, because the
webhook CLI and sandbox both use it. In `mobilerun/`, `core/` holds domain
objects: devices, operators, state, credentials, proxies, numbers, mailboxes,
webhooks, workspace, and extensions. `models/`, `clients/`, `errors/`, `utils/`,
`config/`, and `logs/` supply shared building blocks. Phone behavior lives in
`core/devices/` and its components; account-wide resources live in their own
core folders or on `Workspace`. See the [topic map](core/README.md#topic-map).

| New code | Location |
| --- | --- |
| Device error | `errors/device.py` |
| Macro error | `macro/errors.py` |
| Proxy CLI command | `cli/proxy/` |
| Webhook CLI command | `cli/webhook/` |
| Skills CLI | `skills/cli/` |

## Dependency direction

Name new cloud phones explicitly: `CloudDevice(name="signup-check", billing="paid")`.
Creation options are explicit keywords, not arbitrary SDK passthrough; they are
rejected with an existing device ID. Use `await phone.rename("new-name")` there.
`phone.name` is a read-only cache refreshed by `await phone.info()` or rename.
See [cloud creation options](core/devices/cloud.py).

**Design rule:** Optional resources that work independently bind themselves to a
device. Behavior that only makes sense on a phone lives on the phone.

| Rule | What belongs here | Where each thing goes |
| --- | --- | --- |
| On the phone | Proxy selection/switching; operator selection/execution | `core/devices/cloud.py`, `core/proxy/`, `core/operator/` |
| Binds itself | Phone numbers, mailboxes, credential providers | `core/phone_number/`, `core/mailbox/`, `core/credentials/` |
| Organization-scoped | Webhook endpoints; no device binding | `core/webhooks/` |
| Shared values | App, Billing/Plan, Credential, task/expectation results | `models/`; no SDK, IO or lifecycle |

Use `await number.bind(phone)`, `await mailbox.bind(device)` and
`await provider.bind(device)`. Numbers/mailboxes and providers unbind with
`await x.unbind()`. Device resource state is read-only,
including the ordered `device.credentials` tuple (first-bound wins). Devices do
not accept `credentials=`. Mailbox binding is local association only and works on
local typing devices too. Number binding requires a telephony-capable cloud phone.

Cloud phones always require a proxy; `CloudDevice()` defaults to `CloudRotatingProxy`.
Pass `CloudRotatingProxy`, `CloudFixedProxy` or `CloudSocks5Proxy` explicitly to choose one.
Proxies are extensions: bind one with `await proxy.bind(phone)`. Replace an active
proxy with `await replacement.bind(phone, force=True)`. Operators support constructor and
per-call selection; they are separate from proxy binding.

```text
Each arrow points from the importer to a dependency it may import.

sandbox/, mcp/, macro/, skills/ -> core/, clients/, shared packages
core/                           -> clients/, shared packages
cli/                            -> core/, clients/, shared packages

Shared packages: models/, config/, utils/, errors/, logs/
The root CLI loads feature commands only via mobilerun.cli entry points.
No feature imports another feature or the mobilerun-agents add-on.
```

Shared packages do not import features. Clients do not import core or the root package. Features keep their own CLI, errors, models, settings, and utilities. They may use `cli/base/` and `cli/devices/base/`, but not other shared CLI modules.

## Operators

A device provides actions and state; an operator decides how to execute an instruction. `CloudDevice` defaults to `CloudTaskOperator`. Pass `MacroOperator` for strict replay. The `mobilerun-agents` add-on provides `AgentOperator` for LLM execution and `AgentMacroOperator` for agent fallback on divergence. Pass an operator to the device or override it per `device.do()` call.

## Quickstart

Start or reuse a local AVD with `await AndroidDevice(avd="AVD_NAME")`; see
[device lifecycle](core/devices/README.md#platforms-and-wrappers).

This example requires the unpublished `mobilerun-agents` add-on, an Android device and agent credentials. Running it may consume LLM tokens. For an example using only the base package, see the [device quickstart](../README.md#60-second-quickstart).

```python
import asyncio
from mobilerun.core.devices import AndroidDevice
from mobilerun_agents import AgentOperator  # provided by mobilerun-agents

async def main():
    async with AndroidDevice("emulator-5554", operator=AgentOperator()) as device:
        result = await device.do("Open Settings")
        print(result)
asyncio.run(main())
```

### Cloud quickstart

With cloud API credentials configured (this creates a billable phone):

```python
from pydantic import BaseModel
from mobilerun import CloudDevice
from mobilerun.models import Billing, Plan

class Screen(BaseModel):
    description: str

async def cloud_example():
    cloud = await CloudDevice(name="Quickstart", billing=Billing.PAID, plan=Plan.MINUTE)
    try:
        result = await cloud.do("Describe the current screen", output=Screen)
        return result.output
    finally:
        await cloud.close()
```

Secrets cannot be sent through the public cloud. Create them in the Mobilerun
vault with `CloudCredentials().save(...)`, then attach `CloudCredentials()` to
the cloud device (`await CloudCredentials().bind(cloud)`). Cloud tasks use
vault references, not secret values; local credential providers are rejected
before API calls. A local `AgentOperator` driving a cloud phone remains allowed
to use local credentials.

### Cloud webhooks

With cloud API credentials configured, list your organization's webhook endpoints without creating resources:

```python
from mobilerun.core.webhooks import CloudWebhooks

async def webhook_example():
    for endpoint in await CloudWebhooks(billing="no_commerce").list():
        print(endpoint.id, endpoint.state)
```

Use a registered endpoint ID to [deliver sandbox function results](sandbox/README.md#send-a-function-result-to-a-webhook). Deploying and running that flow can incur charges.

## CLI device selection

For a local example that records `steps.jsonl`, builds a versioned macro, and
loads it for replay, see [Record and replay a macro](../README.md#record-and-replay-a-macro).

DEVICE is optional: run `mobilerun device connect <id>` to save a default, or choose from discovered devices. An explicit DEVICE takes precedence, followed by the saved `device.serial` and discovery. Run `mobilerun device disconnect` to clear the default. Python device APIs remain explicit.

There is no legacy import compatibility layer. With the add-on installed,
import agent code from `mobilerun_agents` directly.
