# Device credentials

`base.py` defines `CredentialProvider`; `dict.py`, `env.py`, `file.py` and
`cloud.py` implement it. The shared `Credential` value type lives in
`mobilerun/models/credentials.py` and is exported by this package.

Credential vaults are extensions. Bind each vault with `await vault.bind(device)`
and detach it with `await vault.unbind()`. Give multiple vaults distinct `name=`
values. Metadata contains package names, credential names and field names, never
values. `type_field` types the value through the device's secret-safe path and
returns `None`.
Before typing, the device refreshes its state and requires a focused editable
field. Providers that cannot report focus and editability, including
screenshot-only providers, reject secret input with `DeviceOperationNotSupported`.

The focus check and secret typing share a task-reentrant input-operation lock on
the root device. Taps, swipes, keys, app changes, ordinary typing and other
focus-changing operations through that root, its wrappers, pools or action catalog
wait until secret input finishes. Wrapper delegation takes the same lock reentrantly.

**Residual risk:** the app or OS can still change focus on its own between the
check and typing. The backend has no atomic "type into this expected field" API.
This lock prevents competing Mobilerun operations, not autonomous app/OS changes
or input from another client/process.

## Quick example

Run inside an async function after focusing an editable field.

```python
from mobilerun.core.credentials import DictCredentials
from mobilerun.core.devices import AndroidDevice

async with AndroidDevice("emulator-5554") as device:
    vault = DictCredentials({"com.example.app": {"work": {"password": "example-only"}}})
    await vault.bind(device)
    await vault.type_field("com.example.app", "password", name="work")
```

Flat mappings and YAML/JSON files automatically become unbound credentials with
one `password` field. `MOBILERUN_SECRET_GMAIL_PASSWORD` becomes name
`GMAIL_PASSWORD` with a password field. A custom prefix is supported.

The full YAML/JSON format is package → credential name → fields:

```yaml
com.google.android.gm:
  work:
    email: example@example.com
    password: example-only
```

Fields match the Cloud vault: `email`, `username`, `password`, `api_token`,
`phone_number`, `two_factor_secret`, `backup_codes`. Package `None` means a
credential is not app-bound. `list(package=None)` lists metadata for all credentials.

Cloud credentials use the same API-key resolution as CloudDevice, including
`mobilerun account login`, unless `api_key=` is explicit. They paginate the vault,
include Mobilerun and 1Password entries, and cache per instance. Call
`await provider.close()` when done. No service authentication storage is changed.

Replace `{{secret:…}}` input with `await vault.type_field(package, field, name)`.

Cloud tasks send only vault references, never secret values. Create credentials in the cloud vault before running a cloud task. Local credentials can be used by a local operator, including on a cloud phone.
Use the vault to manage fields directly (values are never shown in credential
representations):

```python
from mobilerun.core.credentials import CloudCredentials

vault = CloudCredentials()
try:
    await vault.save("com.example.app", "work", {"username": "example", "password": "example-only"})
    await vault.update("com.example.app", "work", "password", "new-example")
    await vault.delete("com.example.app", "work", "username")  # one field
    await vault.delete("com.example.app", "work")  # entire credential
finally:
    await vault.close()
```

`save` creates missing packages and credentials and updates or adds supplied
fields on existing credentials. Writes invalidate the instance cache.

See [devices](../devices/README.md) and [extensions](../extensions/README.md).
