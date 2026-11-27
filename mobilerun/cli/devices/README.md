# devices/

CLI commands and helpers for selecting and controlling devices.

## Quick example

```sh
mobilerun device list
mobilerun device input tap 200 200 emulator-5554
```

Cloud devices can incur charges.

Device actions accept a device ID as the last argument or via `--device`.
Without one, the configured default is used. Otherwise local and authenticated
cloud devices are discovered without creating a device. A single match is used
automatically; multiple matches prompt on a terminal and require an explicit ID
in non-interactive use. Cloud devices that are not ready are marked in the picker.

Run `mobilerun device connect` to choose a default from a picker, pass a
known device ID, or provide a network address such as `host:5555`. The picker
also offers an address-entry option for devices not yet discovered. Run
`mobilerun device disconnect` to clear the default; network defaults are also
disconnected from ADB. Pass a network address to disconnect that device alone.

## Environment controls

```sh
mobilerun device settings location get
mobilerun device settings location get --json
mobilerun device settings location set 52.52 13.40
mobilerun device settings location reset
mobilerun device settings language get
mobilerun device settings language set de-DE
mobilerun device settings timezone get
mobilerun device settings timezone set Europe/Berlin
mobilerun device input clipboard get "$CLOUD_DEVICE_ID"
mobilerun device input clipboard set "hello" "$CLOUD_DEVICE_ID"
mobilerun device app open https://example.com
```

All commands use the normal device selection and accept `[DEVICE]` or `--device`.
Clipboard get and set are cloud-only.
Support varies by backend; see [environment, credentials and proxies](../../core/devices/README.md#environment-credentials-and-proxies).
Emulators retain their last fixed location; reset is unsupported, so set a new one instead.
Reading location is supported on cloud devices; local Android, iOS and remote devices report it as unsupported. The cloud API returns coordinates without a mocked-status flag.

## Command tree

Top-level commands: `list`, `connect`, `disconnect`, `ping`, `setup`.
Subgroups: `app`, `file`, `input`, `screen`, `settings`, `action`, `cloud`.
Package directories mirror these groups, with the group entry in `__init__.py`.
Each selected command accepts `[DEVICE]` or `--device/-d`; conflicting values fail.

| Path | Purpose |
| --- | --- |
| `group.py`, `list.py`, `connect.py`, `disconnect.py`, `ping.py`, `setup.py` | Root device commands |
| [base/](base/README.md) | Shared selection and device options |
| [action/](action/README.md), [app/](app/README.md), [cloud/](cloud/README.md) | Actions, apps and cloud lifecycle |
| [file/](file/README.md), [input/](input/README.md), [screen/](screen/README.md), [settings/](settings/README.md) | File transfer, input, screen and environment controls |

See the [CLI overview](../README.md).

The old flat commands are no longer registered. Use their grouped replacements:

| Old command | Current command |
| --- | --- |
| `device screenshot` | `device screen screenshot` |
| `device ui` | `device screen ui` |
| `device tap` | `device input tap` |
| `device swipe` | `device input swipe` |
| `device long-press` | `device input long-press` |
| `device type` | `device input type` |
| `device press` | `device input press` |
| `device apps` | `device app list` |
| `device start` | `device app start` |

`mobilerun doctor` diagnoses without installing/updating Portal or enabling
accessibility. Use `mobilerun doctor --fix` to permit repairs. Failed checks exit 1;
warnings alone do not fail.
