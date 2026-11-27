# Sandbox cloud functions

Sandbox is a self-contained Mobilerun plugin for deploying typed Python functions
as authenticated endpoints or UTC schedules. Flows can also send results to a
[webhook](#send-a-function-result-to-a-webhook).
Install the `sandbox` extra (`pip install 'mobilerun[sandbox]'`) and configure
cloud credentials before deploying. The extra pins `mobilerun-functions` 0.1.0
with device support.

## Device-free example

Save as `hello.py`:

```python
from pydantic import BaseModel

from mobilerun.sandbox import cloud


class Input(BaseModel):
    name: str = "world"


class Output(BaseModel):
    message: str


@cloud("hello")
def hello(input: Input) -> Output:
    return Output(message=f"hello {input.name}")
```

**Deploying, firing, and enabling can incur charges, even without a device.**
An enabled schedule can incur future charges without your CLI running. The
default billing mode is `paid`; `MOBILERUN_BILLING` changes the default and an
explicit `--billing` or `billing=` overrides it. Deploy, fire, and enable
require `paid`; reads, polling, disable, and owned cleanup allow `no_commerce`.
The CLI asks once before deploying, firing, or enabling with `paid`, including
when retrying after a billing guard rejects `included` or `no_commerce`.
An `enable` batch names all selected flows in one prompt. The default answer
is No. Pass `--yes` to approve without prompting; `--billing paid` alone does
not approve the action in a non-interactive session.

```sh
mobilerun sandbox deploy hello.py --billing paid
mobilerun sandbox fire hello --input '{"name":"world"}' --billing paid
mobilerun sandbox path --json
```

`fire` invokes an endpoint and returns the invocation ID
without waiting. It does not fire cron triggers. Use `mobilerun sandbox list`
to inspect deployments, `mobilerun sandbox runs hello` for recent executions,
and `mobilerun sandbox enable hello` or `mobilerun sandbox disable hello --yes`
to toggle an owned flow. Disabling has its own confirmation, separate from billing.
`fire` accepts `--input-file FILE` instead of `--input`.

Importing a declaration, or executing it as `__main__`, does not itself deploy
anything. An ordinary call to `hello(Input(name="world"))` runs locally. To
deploy from Python, explicitly call `deploy("hello.py", billing="paid")` after
importing `deploy` from `mobilerun.sandbox`. `deploy()`, `remote.fire()`,
`remote(...)`, and `remote.enable()` ask before retrying with `paid` if a guard
rejects lower billing. Without a terminal, pass `billing="paid"`. For
`deploy()` and `remote.enable()` only, `yes=True` also approves a retry
without prompting. Declining or reaching the guard without a terminal or prior
approval raises `BillingRequired` before the operation makes changes.
Deploy checks billing before inspecting the script, so declining does not run
user code during inspection.

**Migration:** `DeploymentPreview`, `SandboxConfirmationRequired`, and
`deploy(confirm=...)` were removed. Run custom confirmation callbacks outside
`deploy()`. A mode that is too low raises `BillingRequired` if not escalated;
`paid` needs no confirmation in the Python API. `remote.fire()` and `remote(...)`
treat `yes` as payload. Pass `billing="paid"` for non-interactive consent on those calls.

## Examples

### Endpoint (default)

Save as `greet.py`:

```python
from pydantic import BaseModel

from mobilerun.sandbox import cloud


class Input(BaseModel):
    name: str


class Output(BaseModel):
    message: str


@cloud("greet")
def greet(input: Input) -> Output:
    return Output(message=f"Hello, {input.name}!")
```

Deploy, then call the endpoint from the CLI or Python:

```sh
mobilerun sandbox deploy greet.py --yes
mobilerun sandbox fire greet --input '{"name":"Ada"}' --yes
```

```python
from mobilerun.sandbox import cloud

run = cloud["greet"].fire(name="Ada")
```

`sandbox deploy` prints the trigger URL. A raw HTTP call sends the input under
`payload`; `MOBILERUN_API_KEY` is the API key. HTTP 202 returns an
`invocationId`, not the result. Poll the printed executions URL using that ID,
then fetch the returned execution ID for `result.steps[].body.result`.

```sh
curl -X POST '<trigger-url>' \
  -H "Authorization: Bearer $MOBILERUN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"payload":{"name":"Ada"}}'
```

### Endpoint that uses a phone

The function receives a runtime `Context`; `ctx.device()` returns its device.
Device time is billed.

```python
from mobilerun_functions import Context
from pydantic import BaseModel
from mobilerun.sandbox import cloud


class Empty(BaseModel):
    pass


class Output(BaseModel):
    message: str


@cloud("open-settings", device="<device-id>")
def open_settings(input: Empty, ctx: Context) -> Output:
    device = ctx.device()
    device.start_app("com.android.settings")
    return Output(message="Settings opened")
```

```sh
mobilerun sandbox deploy open_settings.py --yes
```

Alternatively, leave `device=` off the decorator and deploy with
`mobilerun sandbox deploy open_settings.py --device ID --yes`.

### Scheduled run

This five-field cron runs at 06:00 UTC every day. A schedule can incur future
charges; disable it when it is no longer needed. `fire` does not trigger cron
flows; use `runs` to inspect executions.

```python
from mobilerun_functions import Context
from pydantic import BaseModel
from mobilerun.sandbox import cloud


class Empty(BaseModel):
    pass


class Output(BaseModel):
    message: str


@cloud("nightly-report", schedule="0 6 * * *", device="<device-id>")
def nightly_report(input: Empty, ctx: Context) -> Output:
    device = ctx.device()
    return Output(message=f"Report generated for {device.current_app_id()}")
```

```sh
mobilerun sandbox deploy nightly_report.py --yes
mobilerun sandbox runs nightly-report
mobilerun sandbox disable nightly-report --yes
```

For a device-free schedule, use a separate declaration and a body that does not
access `ctx.device()`:

```python
from pydantic import BaseModel
from mobilerun.sandbox import cloud


class Empty(BaseModel):
    pass


class Output(BaseModel):
    message: str


@cloud("nightly-report", schedule="0 6 * * *")
def nightly_report(input: Empty) -> Output:
    return Output(message="Device-free report generated")
```

```sh
mobilerun sandbox deploy nightly_report.py --device-free --yes
```

### Function only

Set `endpoint=False` to deploy the function without an endpoint or schedule.

```python
from pydantic import BaseModel

from mobilerun.sandbox import cloud


class Input(BaseModel):
    value: str


class Output(BaseModel):
    result: str


@cloud("helper", endpoint=False)
def helper(input: Input) -> Output:
    return Output(result=f"{input.value} is ready")
```

```sh
mobilerun sandbox deploy helper.py --yes
```

## Remote calls

```python
from hello import hello

run = hello.remote.fire(name="world")  # Potentially paid; does not wait.
print(run.invocation_id)
print(run.result().message)  # Validated as Output.
```

`hello.remote(...)` fires and waits in one call. `cloud["hello"]` is a lazy,
untyped handle: it returns plain JSON rather than the declared Pydantic model.
Neither handle deploys on demand. A payload field named `billing` or `input`
belongs inside the explicit `input={...}` mapping. `yes` is a payload keyword,
not billing consent: `remote.fire(yes=True, billing="paid")` sends `{"yes": true}`.
Use either that mapping (or a Pydantic model) or payload keywords, not both.

`Run.result(timeout=300)` shares one deadline across invocation lookup and
execution polling; timing out does not cancel execution. Execution or function
failure raises an error; a partial function result can still return typed
output, with partial metadata in `run.snapshot`. `run.logs()` retrieves logs
separately. Run handles expose execution ID, status, timestamps, and duration.
These Python APIs are synchronous and cannot be called from a running event loop.

## Source, triggers, and devices

- Use one top-level `@cloud(...)` declaration per Python file, with literal
  decorator arguments and the canonical `from mobilerun.sandbox import cloud`
  import (no alias). Keep the typed signature models and any models they
  reference in the file. The packager emits only the function, its required
  models/imports, and the worker runtime wrapper.
- Put helpers, constants, and lazy third-party imports used by the function
  inside its body. Sibling modules, application assets, and arbitrary project
  dependencies are not bundled or installed; unsupported decorators are
  rejected. The deployer parses the original file without importing it, then
  inspects the exact generated source locally. **Only deploy trusted files:**
  inspection executes Python and is not a security boundary.
- `@cloud("name")` deploys an authenticated endpoint by default. To deploy just
  the function without a flow, set `endpoint=False` and omit `schedule`. Use
  `schedule="<five-field UTC cron>"` for a scheduled workflow instead. Explicit
  `endpoint=True` is still supported but cannot be combined with a schedule.
  The client checks endpoint payloads with
  plain `jsonschema`; the server checks `format` constraints. Endpoint input
  schemas that contain `$ref` are rejected rather than resolved or uploaded.
- `--device ID` / `device=` overrides the decorator's device. Use
  `--device-free` / `device_free=True` to override the decorator and omit a
  device. An endpoint without a device defaults to device-free. For a schedule
  without a decorator device, the interactive CLI offers
  existing cloud devices and a device-free choice; noninteractive deployment
  must make that choice explicitly. Sandbox does not provision or delete devices.

## Local store and ownership

`mobilerun sandbox path` shows the expanded absolute store path without
authenticating or creating it. The default is `<MOBILERUN_USER_CONFIG_DIR>/sandbox`.
Set `MOBILERUN_SANDBOX_DIRECTORY` to relocate it; an empty value uses the default.
The store is created on deployment and scoped as follows (`\0` is a NUL separator):

```text
<store>/<sha256(api-origin\0key-fingerprint)>/<sha256(declaration-name)>/
  receipt.json
  flow.yaml
```

The key fingerprint is the SHA-256 of the resolved API key, not the key itself or
a server account ID. Changing the API origin or key selects a different scope.
`receipt.json` records verified creation IDs and deployment progress, including
partial deployments. `flow.yaml` is an inspectable workflow document, not proof
of ownership. No function YAML, source copies, or application assets are stored
there. Preserve the receipt: a lost receipt or key rotation can make previously
deployed resources inaccessible to the plugin's mutation commands. Matching a
name does not grant ownership, and resources are never adopted by name.

Flows pin a numeric function version after it is ready. A replacement schedule
is not activated unless the old schedule's shutdown is confirmed. The cutover
can miss a firing, so do not rely on exactly-once scheduling. If creation is
uncertain, retain the original source and receipt for recovery; do not delete
resources based on names. Cleanup checks recorded IDs and relationships and
preserves foreign/shared resources. Local flow YAML is removed only after
owned flow removal is confirmed.

If a cutover receipt predates this version and lacks journaled action payloads,
the CLI cannot resume or clean up that deployment. Keep the receipt and finish
or clean up the deployment manually in the cloud account. Use the recorded IDs
to identify resources and verify ownership before changing or deleting them.

## Files

| File | Responsibility |
| --- | --- |
| [`__init__.py`](__init__.py), [`_runtime.py`](_runtime.py) | Public API and lazy worker-runtime exports |
| [`app.py`](app.py) | Cloud decorator and synchronous `deploy`, `Remote`, and `Run` APIs |
| [`packaging.py`](packaging.py) | Static declaration parsing, extraction, and local inspection |
| [`deployment.py`](deployment.py) | Async first deploy, redeploy, and workflow cutover |
| [`receipts.py`](receipts.py) | Account scope, the local receipt store, and deploy checkpoints |
| [`documents.py`](documents.py) | Workflow document building and comparison |
| [`flows.py`](flows.py) | Remote function and flow lookup, ownership checks, webhook checks, and enable/disable |
| [`execution.py`](execution.py) | Endpoint runs, run polling, results, and logs |
| [`cleanup.py`](cleanup.py) | Removal of resources recorded in a receipt |
| [`billing.py`](billing.py) | Paid-mode check shared by deploy, run, and activation |
| [`clients.py`](clients.py) | Sandbox's async SDK adapter; uses shared [cloud client and billing helpers](../clients/cloud/README.md), closes clients it creates, and leaves injected clients caller-owned |
| [`models.py`](models.py), [`errors.py`](errors.py), [`settings.py`](settings.py) | Declarations, receipts, errors, and defaults |
| [`cli/`](cli/) | Removable `mobilerun sandbox` command group |

See the [smoke test instructions](../../tests/unit/smoke/README.md#sandbox-cloud-functions)
for the paid, device-free lifecycle check and its cleanup procedure.

## Results and files

### Send a function result to a webhook

[Register an HTTPS receiver](../cli/webhook/README.md), then pass the returned endpoint UUID, not the URL:

```sh
mobilerun webhook register https://receiver.example/results
```

For the [device-free example](#device-free-example), change the decorator to:

```python
@cloud("hello", webhook="<endpoint-id>")
```

Replace `<endpoint-id>` with the actual UUID. If you pass a URL, declaration validation fails with a `mobilerun webhook register <url>` hint. Sandbox does not register or delete webhook endpoints. `webhook=` requires an endpoint or schedule; it cannot be used for a function-only deployment.

**Cost:** Deploying a function and running its flow can incur charges. After a successful function step, the appended `webhooks.send` step queues a signed POST with the return value in `data.result`, for example:

```json
{"type": "workflow.step.emit", "data": {"result": {"message": "hello world"}}}
```

If you also declare `files`, the flow uploads them first and replaces each base64 file value in `data.result` with an object containing `path` and `fileId`. Flow success confirms queuing, not HTTP delivery. Failed functions do not emit a result.

The endpoint must be active, not blocked, and subscribed to `workflow.step.emit` (or all events). Changing the ID updates the flow action, not the endpoint. If it becomes unusable during cutover, redeploy the same function with a usable ID or remove `webhook=`. The deployment keeps its flow ID and retains old actions until the replacement is confirmed.

If `flow.yaml` is missing, enabling a webhook flow checks the live flow against the last completed deployment in the receipt. If that evidence is missing or doesn't match, redeploy before enabling. If cleanup reports an uncertain action create, retry the original deploy before cleaning up so it can recover the action ID.

`run.snapshot.get("webhookStatus")` reports only an observed webhook step: `pending`, `running`, `succeeded`, `failed`, `skipped`, or `cancelled`. No webhook step means no status field. `Run.result()` warns if the webhook failed, was skipped, or was cancelled, but still returns a successful function result. `succeeded` means the event was queued, not that HTTP delivery was confirmed.

### Save images from a function result

Pass `files` to map filenames to dotted result paths and `files_folder` to
choose the destination folder in Mobilerun. This works with a schedule or an
endpoint (including the default endpoint) and can be combined with `webhook=`.

```python
from pydantic import BaseModel, Field

class Event(BaseModel):
    map_screenshot: str

class ReportOutput(BaseModel):
    events: list[Event] = Field(min_length=1, max_length=1)

@cloud(
    "event-report",
    schedule="*/30 * * * *",
    device="<device-id>",
    files={"event_1_map.jpg": "events.0.map_screenshot"},
    files_folder="atc",
)
def report(input: ReportInput, ctx: Context) -> ReportOutput:
    ...
```

The return annotation must be a non-optional Pydantic model. Each path must
use the model's serialization names (`serialization_alias` or `alias` when
set) and point to a required, non-null string containing a base64-encoded image. For a
list index, set `min_length` to at least the index plus one. Scheduled input
must accept `{}`. With `webhook=`, lists on a file path must be fixed-length:
set `min_length` and `max_length` to the same value so the result can keep
its list shape without sending the image data. Objects on file paths cannot
allow extra keys (`extra="allow"`) or have conditionally omitted properties
(`Field(exclude=True)` or `Field(exclude_if=...)`). Ordinary defaulted fields,
including nullable fields defaulting to `None`, are included. Branches without
files are passed through unchanged. Branches rebuilt for the webhook cannot
use `SerializeAsAny` or custom serializers that could change their shape.
Other keys on rebuilt objects can use aliases like `event-title`. Empty keys,
keys with surrounding whitespace, dots or braces, and `__proto__`,
`constructor`, or `prototype` (case-insensitive) are not supported in these
lookups. The dotted paths in `files` use identifier names and array indices.
Filenames must be case-insensitively unique, at most 255
UTF-16 code units, contain no surrounding whitespace, directory separators,
or control characters, and end in `.jpg`, `.jpeg`, or `.png`. The folder must
name at least one segment. Use at
most 1024 UTF-16 code units including the leading `/`, with no empty or `.`
segments, `..` anywhere, or control characters. The example uploads
`event_1_map.jpg` into `/atc` after the function succeeds. With `webhook=`, the
folder and filenames must be literal (no `{{...}}` placeholders), and the
webhook queues only after the uploads succeed. In this example,
`data.result.events[0].map_screenshot` is an object with `path` set to
`/atc/event_1_map.jpg` and `fileId` set to the upload's ID. The `fileId` is
valid when the webhook arrives. The backend writes the Mobilerun folder copy
at `path` only after the run finishes successfully, so the copy may appear
later or not at all if copying fails. For a field uploaded under multiple
filenames, both values refer to the first listed upload.
A webhook failure does not prevent uploads. Deploying and running can incur charges.
