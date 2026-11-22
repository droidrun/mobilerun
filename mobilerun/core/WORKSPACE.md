# Workspace assets

`Workspace` manages account-wide files and apps, independently of a phone.
Every operation is a classmethod; no configured instance is needed.

## Quick example

These operations use a cloud account; uploaded storage or other resources may incur charges.

```python
from mobilerun import Workspace

file = await Workspace.upload_file("photo.jpg")
files = await Workspace.list_files()
path = await Workspace.download_file(file.id, ".")
await Workspace.push_file(file, device, "/sdcard/Download")
await Workspace.delete_file(file.id)

app = await Workspace.upload_app("counter.apk")
apps = await Workspace.list_apps()
app = await Workspace.get_app(app.id)
await device.install_app(app)
await Workspace.delete_app(app.id)
```

Methods accept keyword `api_key=`, `base_url=`, and `billing=` overrides. Defaults
use `MOBILERUN_API_KEY`, then saved login, and `Billing.NO_COMMERCE`. Each operation
owns and closes its client. `push_file` selects the source workspace through
these options; the destination device retains its own account and billing.
Temporary local transfer files are removed even when the destination fails.

Downloads accept an existing directory or a file path. A string ending in `/`
creates a directory. Uploaded app deletion removes files and metadata; it does
not uninstall the app from a phone. Deletions cannot be undone.

See [devices](devices/README.md) and [cloud clients](../clients/cloud/README.md).
