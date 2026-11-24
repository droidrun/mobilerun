# proxy/

CLI proxy commands and helpers.

`proxy release ID` requires confirmation and releases provisioning; phones
using the proxy lose it. Pass `--yes` for automation and `--json` for JSON output.
`proxy buy` and `proxy bind` also support `--json`. Checkout and binding
progress are written to stderr.

Proxy purchases can incur charges.

```sh
mobilerun proxy list --json
```

`proxy release ID...` accepts multiple IDs; `--json` returns an array of per-ID results, including failures.
