# Device settings

`mobilerun device settings` contains `location get/set/reset`, `language get/set`
and `timezone get/set`. Language and timezone getters use the cloud SDK; backends
without support report an unsupported operation. Successful mutations acknowledge
the selected device. Location get accepts `--json`.

```sh
mobilerun device settings location set 52.52 13.40 emulator-5554
```
