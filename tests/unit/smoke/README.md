# Real-world smoke tests

Opt in with `pytest --smoke tests/unit/smoke`; normal `pytest` skips this directory.
Start with `pytest --smoke tests/unit/smoke/test_00_requirements.py -s` for a value-free requirements table.
Use `-m smoke_device` for a local emulator (no service cost), `-m smoke_cloud` for cloud operations, or `-m smoke_llm` for local LLM calls. Cloud operations and tasks may incur charges; local LLM calls may consume provider tokens. Running all smoke tests invokes both. Select individual tests only after reviewing their source and prerequisites.

Set `MOBILERUN_SMOKE_SERIAL` (default `emulator-5554`), `MOBILERUN_SMOKE_CLOUD_DEVICE_ID` (default Tester 1), `MOBILERUN_SMOKE_LLM_PROVIDER` (default `openai_oauth`). Set `MOBILERUN_API_KEY`. The emulator must be online with Mobilerun Portal installed; the cloud device must be ready.

The test harness redirects HOME and the config directory before importing application code. Secrets are restored in memory only for individual smoke tests. OAuth profiles, when needed, are copied read-only from the real config into a temporary directory; the original is never opened for writing. Tests can change device UI and end on the home screen.

TODO: Add local/cloud pool ordering, agent/context provider, skills, sandbox and remaining CLI/macro coverage.

`test_cloud_lifecycle.py::test_cloud_info_and_wait_ready` and
`test_cloud_lifecycle.py::test_cloud_rejects_local_secrets` do not create a phone or run a task; check your account pricing before running cloud operations.
`test_cloud_lifecycle.py::test_paid_cloud_create_lifecycle` creates a billable
phone and resets it; collect only until explicitly approved. The typed-output
task smoke also checks lazy trajectory and screenshot reads.

## Cleanup

Tests can change device UI. Review each test’s `finally` cleanup before running; after interruption, check the cloud dashboard for newly created phones and other resources, and release any that should not be retained. Existing devices are not automatically deleted. Never assume an interrupted run cleaned up.
