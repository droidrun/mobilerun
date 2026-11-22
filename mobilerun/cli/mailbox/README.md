# mailbox/

CLI mailbox commands and helpers.

Creating a rented mailbox costs money; the command below requires confirmation.

## Quick example

```sh
mobilerun mailbox included --json
mobilerun mailbox create --billing paid --plan rent
```

Plans: `auto` uses included quota first; `included` never rents; `rent` requires paid billing.

`mailbox status MAILBOX` shows its address and lifecycle state; `mailbox code
MAILBOX` waits for a verification code. Create, status and release support
`--json`.

## Files

| Path | Purpose |
| --- | --- |
| `group.py` | Mailbox CLI commands |
| `__init__.py` | Package marker |

See the [CLI overview](../README.md).
