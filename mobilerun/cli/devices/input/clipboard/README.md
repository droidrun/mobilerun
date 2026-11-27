# Clipboard commands

On a cloud device, get clipboard text on stdout or set text without echoing the
supplied value. If the selected target is not a cloud device, both commands
report that a cloud device ID is required and exit without connecting.
Bare invocation presents the get/set menu; both `-h` and `--help` show help.

```sh
mobilerun device input clipboard get "$CLOUD_DEVICE_ID"
```
