# Real-world smoke tests

Opt in with `pytest --smoke tests/unit/smoke`; normal `pytest` skips this directory.
Start with `pytest --smoke tests/unit/smoke/test_00_requirements.py -s` for a value-free requirements table.
Use `-m smoke_device` for a local emulator (no service cost), `-m smoke_cloud` for cloud operations, or `-m smoke_llm` for local LLM calls. Cloud operations and tasks may incur charges; local LLM calls may consume provider tokens. Running all smoke tests invokes both. Select individual tests only after reviewing their source and prerequisites.

Set `MOBILERUN_SMOKE_SERIAL` (default `emulator-5554`), `MOBILERUN_SMOKE_CLOUD_DEVICE_ID` (default Tester 1), `MOBILERUN_SMOKE_LLM_PROVIDER` (default `openai_oauth`). Set `MOBILERUN_API_KEY`. The emulator must be online with Mobilerun Portal installed; the cloud device must be ready.

The test harness redirects HOME and the config directory before importing application code. Secrets are restored in memory only for individual smoke tests. OAuth profiles, when needed, are copied read-only from the real config into a temporary directory; the original is never opened for writing. Tests can change device UI and end on the home screen.

TODO: Add local/cloud pool ordering, agent/context provider, skills and remaining CLI/macro coverage.

`test_cloud_devices.py::test_cloud_info_and_wait_ready` and
`test_cloud_devices.py::test_cloud_rejects_local_secrets` do not create a phone or run a task; check your account pricing before running cloud operations.
`test_cloud_devices.py::test_paid_cloud_create_lifecycle` creates a billable
phone and resets it; collect only until explicitly approved. The typed-output
task smoke also checks lazy trajectory and screenshot reads.

## Sandbox cloud functions

The sandbox endpoint smoke requires the `sandbox` and `dev` extras, a valid
`MOBILERUN_API_KEY`, an explicit `MOBILERUN_CLOUD_BASE_URL`, and an account
authorized for paid functions and workflows.
It needs no phone, emulator, or LLM credentials. **Device-free is not free:**
deployment and the function execution may be billed. After explicit approval,
run only this lifecycle check with:

```sh
source ~/.agent-env; .venv/bin/pytest --smoke tests/unit/smoke/test_sandbox_functions.py::test_device_free_sandbox_endpoint -v -s
```

`test_device_free_sandbox_webhook` is a separate paid check; it also needs
`MOBILERUN_SMOKE_WEBHOOK_URL` pointing to a dedicated HTTPS receiver you control.
`test_device_free_sandbox_files` is another paid check. Use a saved-routine-files
capable environment (staging or develop). It verifies a PNG saved to a unique
folder, checks that redeploying without `files` removes delivery, and deletes
the owned deployment and saved files in that folder. Running the whole file
selects all three checks.

To check discovery without calling the cloud:

```sh
source ~/.agent-env; .venv/bin/pytest tests/unit/smoke/test_sandbox_functions.py --collect-only -q
```

The test creates one UUID-named function, a ready version, and an endpoint flow
with a trigger and function action. It fires once, polls for a typed pong, and
prints resource and run IDs, available timing, and cleanup outcome (not keys,
payloads, or URLs). Check account billing for actual usage and cost.

The test prints the temporary receipt path before deploying. Its `finally`
block uses [`mobilerun.sandbox.deployment`](../../../mobilerun/sandbox/deployment.py)
to load that receipt even after a partial deployment, disable/remove the
owned flow, and delete the unshared owned trigger, action, and function in
`no_commerce` mode. It then reads the cloud state to verify removal. If the run
is interrupted or cleanup fails, preserve the printed receipt and test source
before pytest prunes its temporary files. Inspect the UUID name and recorded
IDs in the account used for the test. If a create response was lost, cleanup
can refuse because not all IDs are known; review the journal with an operator
before authorizing recovery. Never delete a resource solely because its name
matches, and do not assume re-running pytest will clean up a prior UUID.

## Cleanup

Tests can change device UI. Review each test’s `finally` cleanup before running; after interruption, check the cloud dashboard for newly created phones and other resources, and release any that should not be retained. Existing devices are not automatically deleted. Never assume an interrupted run cleaned up.
