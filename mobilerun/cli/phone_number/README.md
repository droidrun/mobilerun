# phone_number/

CLI phone number commands and helpers.

Renting a phone number costs money; the create command below requires confirmation.
Sending SMS can also incur charges.

## Quick example

```sh
mobilerun number included --json
mobilerun number included --country US --json
mobilerun number create --country US --billing paid --plan rent
mobilerun number bind NUMBER_ID DEVICE_ID
mobilerun number unbind NUMBER_ID
```

Plans: `auto` uses included quota first; `included` never rents; `rent` requires paid billing.
Without `--country`, `number included` lists the remaining included quota for all
countries. Interactive `number create` shows the included quota alongside each
in-stock country; quota is a snapshot, not a reservation.
If a country's quota cannot be fetched, it is shown as unavailable and named
in a warning. The JSON result uses `null` for that country's remaining count.

`number status NUMBER` shows the number, lifecycle state and bound device.
`number messages NUMBER` lists SMS history; `number countries` lists supported
countries. `number send NUMBER TO TEXT --billing paid` requires confirmation and
may incur charges. `number code NUMBER` waits for a verification code. Create, status, release, messages and countries
support `--json`.

## Files

| Path | Purpose |
| --- | --- |
| `group.py` | Phone-number CLI commands |
| `__init__.py` | Package marker |

See the [CLI overview](../README.md).

`number release ID...` and `number unbind ID...` accept multiple IDs, confirm once, and return arrays of per-ID results with `--json`. Use `--yes` in automation.
