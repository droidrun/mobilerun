# Cloud device lifecycle

`mobilerun device cloud` exposes status, resume, stop, reboot and terminate for an
existing cloud phone. These operations call the billing-aware SDK boundary without
opening a device connection, so stopped phones can be resumed or inspected.
Terminate permanently deletes the phone and requires confirmation (`--yes` in
automation). Missing IDs offer the cloud-device picker on a terminal.

Cloud phone operations can incur charges.

```sh
mobilerun device cloud status
```
