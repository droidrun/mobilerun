# Devices

`Device` is the common action interface. Construct a platform device, then call
`connect()`, await it, or use an async context. These all perform the same
idempotent setup. Context exit calls `close()`.

```python
from mobilerun import AndroidDevice, RecordingDevice
from mobilerun.core.state import StateProviderOptions

async with AndroidDevice("emulator-5554", state_options=StateProviderOptions(vision=True)) as phone:
    device = RecordingDevice(phone)
    state = await device.state.refresh()
    print(state.formatted_text)
    assert device.state.current is state
    await device.tap_element(0)
    await device.wait(1)
    print(device.recorded_actions)
```

## Quick example

Connect a local Android device or running emulator. Run this inside an async function.

```python
from pathlib import Path
from mobilerun import AndroidDevice

async with AndroidDevice("emulator-5554") as phone:
    await phone.tap(200, 400)
    await phone.input_text("hello")
    await phone.swipe(200, 700, 200, 300, duration_ms=500)
    Path("screen.png").write_bytes(await phone.screenshot())
    print((await phone.state.refresh()).formatted_text)
```

See [state](../state/README.md), [credentials](../credentials/README.md),
[proxies](../proxy/README.md), and [Workspace](../WORKSPACE.md).

## State and operators

`device.state` is a controller with only `refresh()`, `current`, and
`capabilities`. `refresh()` returns a `UIState`; `current` is that cached snapshot
and raises `DeviceNotReady` until the first refresh. Read elements, text, and
coordinate helpers from the snapshot, not from the controller.

Configure state at construction using `state_options=`, a `state_provider=`
instance, or a `state_provider_factory=(device, options) -> StateProvider`.
An instance and factory are mutually exclusive. Built-in providers and factories
are lazily bound to the outer wrapper so observations see its behavior. An
explicit provider instance is reused as supplied.

```python
from mobilerun import AndroidDevice
# Requires the unpublished mobilerun-agents add-on.
from mobilerun_agents import AgentOperator

device = AndroidDevice("emulator-5554", operator=AgentOperator())
await device.connect()
await device.do("Open Settings")
await device.expect("Settings is visible")
# Override only this operation:
await device.do("Read the battery level", operator=AgentOperator())
```

Providers, operators, raw trees/time, coordinate calibration, recovery, unwrap,
and backend handles are internal machinery. There are no runtime attach methods
or compatibility aliases. Advanced raw-driver composition lives in
`components/wrapper.py`, not the root exports.

## Platforms and wrappers

- `AndroidDevice(serial)` uses a connected ADB phone or emulator.
- `await AndroidDevice(avd="Pixel_API_35", headless=True)` starts or reuses an
  emulator. Only an emulator started by that instance is stopped on close;
  `keep=True` retains it. `timeout=` controls boot wait.
- `CloudDevice(device_id=...)` uses an existing phone. Awaiting a device without
  an ID creates one and can incur charges. `keep=True` retains created phones.
- `IOSDevice(url, token=...)` detects the Portal protocol once during connect.
  Legacy Portal uses the iOS text-tree provider; Portal HTTP uses the Android
  tree provider with point-space coordinates.
- `RemoteDevice(url=...)` uses screenshot-only state by default.
- `RecordingDevice(device)` records actions in `.recorded_actions`.
- `StealthDevice(device)` applies humanized interaction behavior.

Wrappers explicitly delegate operations and private hooks. They do not forward
unknown attributes. `do()` and `expect()` execute against the outer wrapper.
Keep a typed platform handle for platform-only operations, such as cloud rename.
`device.supported_operations` describes backend operations; presence of a method does
not guarantee backend support. Unsupported actions raise
`DeviceOperationNotSupported`. `supported_buttons` lists button names.

Focus-changing input operations share one lock on the root device, including
calls through wrappers, pools and action catalogs. Secret input holds it across
the focused-field precheck and raw typing. Base methods, subclass overrides and
device mixins are instrumented automatically. Wrapper delegation is reentrant
only within the same asyncio task; child tasks wait like any competing operation.
Direct driver access and autonomous app/OS focus changes are outside this lock.

## Installed apps versus workspace apps

```python
from mobilerun import Workspace

app = await Workspace.upload_app("counter.apk")
await cloud_device.install_app(app)
await cloud_device.install_app(await Workspace.get_app("app-id"))
await cloud_device.uninstall_app(app.package)
await Workspace.delete_app(app.id)  # deletes uploaded files/metadata, not an install

local_app = await local_device.install_apk("counter.apk", reinstall=True)
await local_device.start_app(local_app.package)
apps = await local_device.list_apps()  # user apps only
all_apps = await local_device.list_apps(include_system=True)
```

`install_apk(path)` and `install_app(App)` stay separate. Cloud path installation
uploads using the destination device's credentials. Cloud installs wait for
success and the requested version to appear; pass `timeout=` or set
`MOBILERUN_CLOUD_APP_INSTALL_TIMEOUT_SECONDS` (default 180).
Local Android verifies the installed package. Portal HTTP and local iOS may not
support installation/uninstallation.

## Files and pools

```python
from mobilerun import DevicePool, Workspace

file = await Workspace.upload_file("photo.jpg")
await Workspace.push_file(file, device, "/sdcard/Download")
await device.push_file("local.txt")
files = await device.list_files("/sdcard/Download")
await device.pull_file(files[0].path, "downloaded.txt")

async with DevicePool([first, second], concurrency=2) as pool:
    states = await pool.state.refresh()
    await pool.start_app("com.android.settings")
```

Pool results preserve device order. Package methods broadcast a single package
string. For different packages, iterate `pool.devices` explicitly. Pull files
from members individually with distinct local destinations; pools have no
`pull_file` method. Pools do not expose cloud lifecycle methods.

Android image/video pushes default to `/sdcard/DCIM/Camera` and trigger a media
scan; other files default to `/sdcard/Download`. Cloud uploads do not expose
media scanning. Local iOS and remote devices may not support files.

## Environment, credentials, and proxies

| Value | Definition |
| --- | --- |
| `Location` | `mobilerun.models.location` — device latitude and longitude. |

Location, timezone, URL, language, and clipboard operations are declared on
`Device` and delegated by wrappers and pools. Cloud implements these through its
API. Android emulator location uses `adb emu geo fix`; Android timezone and URL
use shell commands. Unsupported backend combinations fail explicitly.

```python
location = await device.get_location()  # Cloud: latitude/longitude, or None
locations = await pool.get_location()  # One result per device, in order
await device.set_language("de-DE")
locale = await device.get_language()  # Cloud BCP-47 locale
timezone = await device.get_timezone()  # Cloud IANA timezone, or None
await device.set_clipboard("hello")
text = await device.get_clipboard()
await device.open_url("https://example.com")
```

Reading location is unsupported on local Android, iOS and remote devices. The
cloud API does not expose a mocked-status flag.

Bind credentials and communications from the resource side:
`await provider.bind(device)`, `await number.bind(device)`, and
`await mailbox.bind(device)`. Credentials are a read-only tuple.
`device.number` and `device.mailbox` return typed resources; unavailable resources
raise `DeviceOperationNotSupported`. An `AgentOperator` from the optional `mobilerun-agents` package can use local
credentials on a cloud phone; a `CloudTaskOperator` requires cloud vault references.
Only reference names, never secret values, are sent to public cloud tasks.

Cloud credentials resolve from `MOBILERUN_API_KEY`, then saved login. There is no
legacy environment fallback. Device billing follows settings; Workspace defaults
to `Billing.NO_COMMERCE`. Creation and lifecycle operations can incur charges or
delete data.

Creating a cloud phone and rotating proxy can incur charges.

```python
from mobilerun import CloudDevice
from mobilerun.core.proxy import CloudRotatingProxy

proxy = CloudRotatingProxy(country="DE")
async with CloudDevice(country="DE", proxy=proxy,
                       name="signup-check", billing="paid", plan="minute") as phone:
    await phone.info()
    await phone.rename("signup-complete")
    status = await proxy.status()
```

Cloud creation requires a proxy. Omission selects configured rotating-proxy
defaults; `None` is invalid. `CloudFixedProxy` reuses purchased capacity;
`CloudSocks5Proxy` uses external credentials. Android supports `LocalProxy`.

| Class | Where it works | What it does |
| --- | --- | --- |
| `CloudRotatingProxy` | Cloud phones | Uses rotating residential Connect capacity |
| `CloudFixedProxy` | Cloud phones | Reuses a purchased Connect proxy |
| `CloudSocks5Proxy` | Cloud phones | Uses your own SOCKS5 server |
| `LocalProxy` | Local Android | Sets a plain HTTP proxy without authentication |
Proxies are extensions: `await proxy.bind(device)` and `await proxy.status()`.
`phone.proxy` returns the extension object; retain the original handle for static
typing. There is no typed lookup API. Cloud creation publishes the
exact supplied/default proxy instance without a second provisioning call. Cloud
phones reject standalone proxy unbind: use `await replacement.bind(phone,
force=True)` to switch. Local proxies may be unbound normally. Closing a borrowed
cloud phone detaches the local extension without disconnecting its server proxy.
Keep passwords in the environment or a vault.

Connecting an existing cloud phone adopts its remote proxy without SDK mutations:
known Connect IDs become `CloudFixedProxy`; connected but unidentified proxies
become typed `ExistingCloudProxy` observations. Both expose `status()` and require
`force=True` replacement. If the phone is disconnected, no extension is published;
`phone.proxy` raises an error suggesting `await proxy.bind(phone)`. Nothing is
automatically provisioned. Failed/ cancelled binds and failed rollbacks read remote
status before re-raising, so the registry reflects the actual active proxy. Only
unreadable remote state leaves a binding uncertain and blocks ordinary actions.

Creation-only options include `name`, `country`, `locale`, `timezone`,
`android_version`, `apps`, `template`, `proxy`, `device_type`, and `plan`.
Do not combine `template` with `android_version` or `apps`. Existing IDs reject
creation options. `phone.name` is a read-only response cache; `info()` refreshes
it. Existing IDs are never automatically terminated. Cloud-only lifecycle
methods include `resume`, `stop`, `wait_ready`, `reboot`, `reset`, and `terminate`.

For iOS portal authentication, pass credentials using `token=` rather than embedding them in the URL.

### Migrating placeholder input

Placeholder expansion has been removed. Replace `{{secret:…}}` with
`await vault.type_field(package, field, name)` after binding the vault. Replace
`{{code:sms}}` with `code = await sms.wait_for_code()` followed by
`await device.input_text(code)`. Codes are ordinary values; credential values
are typed directly by the vault and are not returned.
