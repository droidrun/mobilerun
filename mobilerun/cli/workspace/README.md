# Workspace CLI

`mobilerun workspace files` and `mobilerun workspace apps` each offer `list`,
`upload`, and removal. Files use `rm` (with a hidden `remove` alias); apps use `remove`. File downloads use `files download FILE_ID [DEST]`. Removal requires interactive confirmation or `--yes`.
Use `mobilerun device file push --file-id <id>` and `mobilerun device app install
--app-id <id>` to put workspace items on devices.

```sh
mobilerun workspace files list --json
mobilerun workspace apps list --json
```

Cloud storage and devices can incur charges. See the [CLI overview](../README.md).

File upload accepts multiple paths; with `--json` it returns an array of uploaded
files. File download and removal, and app upload and removal, also accept `--json`.
Progress and confirmations are written to stderr.
