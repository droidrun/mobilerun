# Clipboard commands

On a cloud device, get clipboard text on stdout or set text without echoing the
supplied value. Local Android and iOS devices do not support clipboard access.
Bare invocation presents the get/set menu; both `-h` and `--help` show help.

```sh
mobilerun device input clipboard get "$CLOUD_DEVICE_ID"
```
