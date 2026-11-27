# Device CLI base

Shared device selection and options.

`@device_command` adds `[DEVICE]`, `--device/-d`, configuration, device-kind and
transport options. Command handlers receive one `DeviceTarget`; use
`async with target.open() as device:` to connect and guarantee cleanup.
Optional targets can represent no device (doctor). Lifecycle commands validate a
cloud-only target and use the cloud client directly rather than opening a phone.

Direct commands do not auto-install Portal. Agent run and macro replay explicitly
preserve their configured auto-setup behavior with `allow_auto_setup=True`.

```sh
mobilerun device ping emulator-5554
```
