# Device actions

`mobilerun device action list --json` describes public callable actions.
`call ACTION --arg key=value` validates and converts schema arguments, then invokes
the action. Interactive calls can choose an action or enter its name manually.
`extensions --json` lists bound extensions and their current context.

Sending through a bound CloudPhoneNumber uses that extension's billing mode.
In `paid` mode it does not ask; if the SDK requires `paid`, `--yes` authorizes
one retry of the SMS send. Sending SMS may incur charges.

```sh
mobilerun device action list --json
```
