# sandbox/cli/

Exposes `mobilerun sandbox` commands to deploy typed cloud functions, invoke
them, and manage their workflows. Start with `mobilerun sandbox --help` or
`mobilerun sandbox path --json`; neither calls the cloud.

| Command | Purpose |
| --- | --- |
| `deploy FILE` | Deploy a `@cloud` function as an authenticated endpoint by default. |
| `list` | List all functions visible in the account, including functions not owned by this plugin. |
| `fire NAME` | Invoke an endpoint without waiting. |
| `runs NAME` | Show recent workflow executions. |
| `enable [NAME]...` | Enable one or more owned workflows. |
| `disable [NAME]...` | Disable one or more owned workflows. |
| `path` | Show the local sandbox store path. |

Deploying, firing, and enabling can incur charges, even without a device.
These commands ask once before proceeding with paid billing; `enable` asks for
all selected workflows together. Pass `--yes` to approve without prompting.
Non-interactive runs require `--yes`, even with `--billing paid`.
Appearing in `list` does not establish ownership or authorize this plugin to mutate
a function or its workflow. Mutations require verified local creation receipts.
Set `endpoint=False` without a schedule to deploy without a flow, or use `schedule=`
to deploy a scheduled workflow instead of an endpoint.
See the [sandbox guide](../README.md) for setup, billing, and examples.

| File | Responsibility |
| --- | --- |
| `__init__.py` | Sandbox CLI package marker. |
| `group.py` | Click command group and local `path` command. |
| `commands.py` | Cloud command options, prompts, and output. |
