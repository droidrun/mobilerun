# Device input

`mobilerun device input` contains tap, long-press, swipe, type and press commands.
Clipboard commands are cloud-only. `clipboard get` prints clipboard text;
`clipboard set TEXT` acknowledges without printing the text. Bare clipboard
opens its menu on a terminal or shows help.

```sh
mobilerun device input tap 200 200 emulator-5554
mobilerun device input clipboard get "$CLOUD_DEVICE_ID"
mobilerun device input clipboard set "hello" "$CLOUD_DEVICE_ID"
```
