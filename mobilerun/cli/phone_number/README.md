# phone_number/

CLI phone number commands and helpers.

Renting a phone number costs money. Sending SMS can also incur charges.

## Quick example

```sh
mobilerun number included --json
mobilerun number included --country US --json
mobilerun number create --country US --billing paid --plan rent
mobilerun number bind NUMBER_ID DEVICE_ID
mobilerun number unbind NUMBER_ID
```

Plans: `auto` uses included quota first; `included` never rents; `rent` requires paid billing.
In `paid` mode, `auto` and `rent` ask for confirmation because they can incur
charges. `included` does not require paid confirmation. If a billing guard
rejects the mode, the command can retry once after consent. Pass `--yes` to approve
without prompting; it is required in non-interactive shells when confirmation
is needed. Exhausted included quota is not retried.
Without `--country`, `number included` lists the remaining included quota for all
countries. Interactive `number create` shows the included quota alongside each
in-stock country; quota is a snapshot, not a reservation.
If a country's quota cannot be fetched, it is shown as unavailable and named
in a warning. The JSON result uses `null` for that country's remaining count.

`number status NUMBER` shows the number, lifecycle state and bound device.
`number messages NUMBER` lists SMS history; `number countries` lists supported
countries. `number send NUMBER TO TEXT` may incur charges and requires
confirmation even with `--billing paid`. Use `--yes` to confirm without
prompting; it is required in non-interactive shells. `number code NUMBER` waits
for a verification code. Create, status, release, messages and countries support
`--json`.

## Files

| Path | Purpose |
| --- | --- |
| `group.py` | Phone-number CLI commands |
| `__init__.py` | Package marker |

See the [CLI overview](../README.md).

`number release ID...` and `number unbind ID...` accept multiple IDs, confirm once, and return arrays of per-ID results with `--json`. Use `--yes` in automation.
