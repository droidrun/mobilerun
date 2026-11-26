# Device actions

`mobilerun device action list --json` describes public callable actions.
`call ACTION --arg key=value` validates and converts schema arguments, then invokes
the action. Interactive calls can choose an action or enter its name manually.
`extensions --json` lists bound extensions and their current context.

Sending through a bound CloudPhoneNumber uses that extension's billing mode.
Sending SMS may incur charges and requires confirmation, even in `paid` mode.
If the send must retry as `paid`, it asks only once. Use `--yes` to confirm
without prompting; it is required in non-interactive shells.

```sh
mobilerun device action list --json
```
