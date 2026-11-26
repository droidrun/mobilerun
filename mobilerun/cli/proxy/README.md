# proxy/

CLI proxy commands and helpers.

`proxy release ID` requires confirmation and releases provisioning; phones
using the proxy lose it. Pass `--yes` for automation and `--json` for JSON output.
`proxy buy` and `proxy bind` also support `--json`. Checkout and binding
progress are written to stderr.

Proxy purchases can incur charges. `proxy buy --billing paid` allows charges
without a billing prompt. In `included` or `no_commerce`, the command asks before
retrying once with `paid`. Use `--yes` to approve that retry in automation.

```sh
mobilerun proxy list --json
mobilerun proxy list --included
mobilerun proxy list --included --json
```

Choose `included` in the interactive `mobilerun proxy` menu, or run `mobilerun
proxy list --included`. Both show only existing proxies billed as included, not
remaining plan quota. The table displays billing mode when known.

`proxy release ID...` accepts multiple IDs; `--json` returns an array of per-ID results, including failures.
