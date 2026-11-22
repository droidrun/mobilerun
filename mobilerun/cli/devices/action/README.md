# Device actions

`mobilerun device action list --json` describes public callable actions.
`call ACTION --arg key=value` validates and converts schema arguments, then invokes
the action. Interactive calls can choose an action or enter its name manually.
`extensions --json` lists bound extensions and their current context.

```sh
mobilerun device action list --json
```
