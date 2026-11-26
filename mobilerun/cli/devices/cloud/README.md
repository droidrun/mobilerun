# Cloud device lifecycle

`mobilerun device cloud` exposes status, resume, stop, reboot, rename and terminate for an
existing cloud phone. These operations call the billing-aware SDK boundary without
opening a device connection, so stopped phones can be resumed or inspected.
Terminate permanently deletes the phone and requires confirmation (`--yes` in
automation). Missing IDs offer the cloud-device picker on a terminal.

`resume` checks the phone's reported billing strategy once. It does not ask for
confirmation when the strategy is `subscription`. A minute, auto, or unknown
strategy requires confirmation even in `paid` mode. In a restricted mode it
asks once before retrying with `paid`. Use `--yes` to confirm without prompting;
it is required for a potentially metered resume in non-interactive shells.

```sh
mobilerun device cloud status
mobilerun device cloud rename DEVICE "My phone"
```
