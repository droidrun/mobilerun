# Device files

`mobilerun device file list` lists a directory. `push` accepts a local file or
`--file-id` from the workspace. `pull` downloads a device file and `rm` deletes one
after confirmation. Transfer progress goes to stderr; paths and JSON stay on stdout.

```sh
mobilerun device file list /sdcard emulator-5554
```
