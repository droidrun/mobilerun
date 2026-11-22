# mobilerun/

Mobilerun connects device interfaces, execution operators, clients, and state providers. Its shared framework supports deterministic macros, sandbox apps, and coding-agent skills. LLM-backed execution and fallback are provided by the separate `mobilerun-agents` add-on (`pip install mobilerun-agents`).

Pause with `await device.wait(seconds)`, not `asyncio.sleep()`: RecordingDevice
records device waits so macro replay keeps the pause.

See the [demos](../tests/demos/README.md) for source-checkout examples and costs.

## Package map

| Folder | Purpose |
| --- | --- |
| [addons/](addons/README.md) | Optional add-on discovery. |
| [core/](core/README.md) | Device and state interfaces. |
| [models/](models/) | Shared app, billing, credential and result value types; no IO or SDK. |
| [core/devices/](core/devices/README.md) | Device implementations. |
| [core/devices/base.py](core/devices/base.py) | Device protocol and shared mixin. |
| [core/devices/cloud.py](core/devices/cloud.py) | Cloud device, lifecycle, apps and task operator. |
| [core/devices/components/](core/devices/components/) | Driver, recording and stealth facades. |
| [core/phone_number/](core/phone_number/README.md) | Phone number interface and cloud implementation. |
| [core/mailbox/](core/mailbox/README.md) | Mailbox interface and cloud implementation. |
| [core/proxy/](core/proxy/README.md) | Proxy interface, Connect and SOCKS5 implementations. |
| [core/state/](core/state/README.md) | UI state providers and models. |
| [core/operator/](core/operator/README.md) | Operator interface. |
| [clients/](clients/README.md) | Shared service clients. |
| [clients/cloud/](clients/cloud/README.md) | Cloud account access. |
| [clients/portal/](clients/portal/README.md) | Portal client support. |
| [clients/local/](clients/local/README.md) | Functional local adapters, APK inspection and code polling. |
| [config/](config/README.md) | Shared configuration. |
| [core/credentials/](core/credentials/README.md) | Device-owned credential providers. |
| [utils/](utils/README.md) | Shared helpers. |
| [errors/](errors/README.md) | Shared errors. |
| [logs/](logs/README.md) | Shared logging. |
| [diagnostics/](diagnostics/README.md) | Diagnostics. |
| [assets/](assets/README.md) | Shared bundled assets. |
| [cli/](cli/README.md) | Root CLI and shared CLI helpers. |
| [macro/](macro/README.md) | Recording and guarded replay. |
| [sandbox/](sandbox/README.md) | Sandbox apps and tasks. |
| [skills/](skills/README.md) | Bundled coding-agent skills. |
| `mobilerun-agents` (separate package) | Optional LLM agents, integrations and agent commands (`pip install mobilerun-agents`). |

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
shared base: models, core, clients, config, utils, errors, logs
    |             |              |
    v             v              v
  macro/       sandbox/        skills/    (independent of one another)
    \             |              /
     +------> mobilerun-agents <-----+

cli/app.py registers feature CLIs.
```

Arrows show permitted use, not a requirement to import every package. Shared base packages do not import features. Features do not import each other or the optional agent add-on. Feature CLI code may import the shared `cli/base/`; `cli/app.py` is a boundary-test exception.

## Operators

A device provides actions and state; an operator decides how to execute an instruction. `CloudDevice` defaults to `CloudTaskOperator`. Pass `MacroOperator` for strict replay. Install `mobilerun-agents` to use `AgentOperator` for LLM execution or `AgentMacroOperator` for agent fallback on divergence. Pass an operator to the device or override it per `device.do()` call.

## Quickstart

Start or reuse a local AVD with `await AndroidDevice(avd="AVD_NAME")`; see
[device lifecycle](core/devices/README.md#platforms-and-wrappers).

Install `mobilerun-agents` (`pip install mobilerun-agents`) first. With an Android device available and agent credentials configured:

```python
import asyncio
from mobilerun.core.devices import AndroidDevice
from mobilerun.agents import AgentOperator  # provided by mobilerun-agents

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

## CLI device selection

DEVICE is optional: run `mobilerun device connect <id>` to save a default, or choose from discovered devices. An explicit DEVICE takes precedence, followed by the saved `device.serial` and discovery. Run `mobilerun device disconnect` to clear the default. Python device APIs remain explicit.
