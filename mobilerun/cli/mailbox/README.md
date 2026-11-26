# mailbox/

CLI mailbox commands and helpers.

Creating a rented mailbox costs money.

## Quick example

```sh
mobilerun mailbox included --json
mobilerun mailbox create --billing paid --plan rent
```

Plans: `auto` uses included quota first; `included` never rents; `rent` requires paid billing.
In `paid` mode, `auto` and `rent` ask for confirmation because they can incur
charges. `included` does not require paid confirmation. If a billing guard
rejects the mode, the command can retry once after consent. Pass `--yes` to approve
without prompting; it is required in non-interactive shells when confirmation
is needed. Exhausted included quota is not retried.

`mailbox status MAILBOX` shows its address and lifecycle state; `mailbox code
MAILBOX` waits for a verification code. Create, status and release support `--json`.

`mailbox messages MAILBOX` lists received emails, and `--json` returns the full
message summaries. The API reports whether a message contains an OTP, but its
history response does not include the code itself.

## Files

| Path | Purpose |
| --- | --- |
| `group.py` | Mailbox CLI commands |
| `__init__.py` | Package marker |

See the [CLI overview](../README.md).

`mailbox release ID...` accepts multiple IDs and returns an array of per-ID results with `--json`.
