# Coding agent guide

Mobilerun is a Python framework for controlling local and cloud mobile devices.
It provides a CLI, deterministic macros and coding-agent skills. LLM agent
execution is provided by the separate `mobilerun-agents` add-on.

## Agent skills

For the framework design and usage guide, see
[`skills/mobilerun-framework/SKILL.md`](skills/mobilerun-framework/SKILL.md).
Agent skills live in the `skills/` submodule (`droidrun/skills`).

## Repository map

See [mobilerun/README.md](mobilerun/README.md) for the detailed package map,
examples, and dependency diagram.

- `mobilerun/core/`: device, state, operator and resource interfaces.
- `mobilerun/core/webhooks/`: organization webhook endpoints; commands in `mobilerun/cli/webhook/`.
- LLM-backed agent execution is in the separate `mobilerun-agents` package (not yet publicly published).
- `mobilerun/clients/`: cloud, local and Portal service adapters.
- `mobilerun/models/`, `config/`, `errors/`, `logs/`, `utils/`: shared types and support.
- `mobilerun/cli/`: command groups and shared CLI helpers.
- `mobilerun/macro/`, `sandbox/`, `skills/`: independent framework features.
- `mobilerun-agents` (separate package): agent execution, integrations and agent commands.
- `tests/unit/`: unit tests and opt-in real-world smoke tests.
- `skills/`: git submodule; `hatch_build.py` bundles its files in the wheel.

Shared core must not import agents or feature packages. The intended boundaries
are checked in `tests/unit/architecture/test_layers.py`. The agent-required
dependency test (`tests/unit/test_agent_required_dependencies.py`) lives in the
separate `mobilerun-agents` repository. The CLI entry point has a special role; see the boundary tests for exceptions.
The v008 migration and legacy path resolver locate `mobilerun_agents` prompt assets without importing the add-on.

## Where code goes

**Could someone delete this folder and still use the rest of Mobilerun?**
If yes, it belongs in an optional feature (`sandbox/`,
`mcp/`, `macro/`, `skills/`); if not, put it in `core/` or a shared package.
For example, `sandbox/` is removable; `core/webhooks/` is not, because the
webhook CLI and sandbox both use it. In `mobilerun/`, `core/` holds domain
objects: devices, operators, state, credentials, proxies, numbers, mailboxes,
webhooks, workspace, and extensions. `models/`, `clients/`, `errors/`, `utils/`,
`config/`, and `logs/` supply shared building blocks. Phone behavior lives in
`core/devices/` and its components; account-wide resources live in their own
core folders or on `Workspace`. See the [topic map](mobilerun/core/README.md#topic-map).

Features keep their own CLI, errors, models, settings, and utilities in their
folders. Shared packages (including `cli/`) do not import features. Features do
not import each other or the agent add-on. Core can use clients; clients do not
import core or the root package. Register feature CLI groups via
`mobilerun.cli` entry points in `pyproject.toml`.

Core subpackages (`core/credentials/`, `core/devices/`, `core/webhooks/`, …)
are part of core, not features: they never get their own `settings.py`,
`utils.py`, `errors.py` or `models.py`. Their tunables go in
`config/settings.py`, helpers in `utils/`, errors in `errors/`, models in
`models/`. Domain constants that callers need (e.g. a default event type) are
exported from the subpackage's `__init__.py`; backend limits stay private in
the module that uses them. Only feature packages (`macro/`, `mcp/`, `skills/`,
`sandbox/`, `addons/`) own a `settings.py`.

| New code | Location |
| --- | --- |
| Device error | `errors/device.py` |
| Macro error | `macro/errors.py` |
| Core tunable (e.g. webhook page size) | `config/settings.py` |
| Core helper (e.g. URL masking) | `utils/` (e.g. `utils/redaction.py`) |
| Core cloud adapter | `clients/cloud/<area>.py` |
| Proxy CLI command | `cli/proxy/` |
| Webhook CLI command | `cli/webhook/` |
| Skills CLI | `skills/cli/` |

## Setup

Use Python 3.11–3.13. From the repository root:

```bash
git submodule update --init
python -m venv .venv
source .venv/bin/activate
pip install -e '.[sandbox,dev]'
```

The equivalent uv-based setup is `uv venv` followed by
`uv pip install -e '.[sandbox,dev]'`. Activate the environment before using the commands
below, or invoke tools directly from `.venv/bin/`.

## Tests and checks

```bash
# Fast unit suite; smoke tests are skipped unless explicitly opted in.
pytest tests/unit

# One test file or one test.
pytest tests/unit/addons/test_plugins.py
pytest tests/unit/architecture/test_layers.py::test_shared_packages_do_not_import_features

# Ruff lint; Black formatting and the CI formatting check.
ruff check .
black .
black --check --diff --verbose .

# Type checking; mypy is configured in pyproject.toml.
mypy mobilerun
```

Black 26.10.1 runs in CI. Ruff is a separate local check. Pyright is not
configured or installed by the project's development dependencies; if available,
`pyright mobilerun` is an additional check, not the documented CI gate.

Smoke tests are opt-in and can use real devices, cloud resources and LLM tokens.
**Ask the user before running any paid smoke test, cloud operation or LLM call.**
Review [smoke setup, costs and cleanup](tests/unit/smoke/README.md) first.
Webhook smoke tests use `webhook.site`; paid sandbox runs delete the webhook endpoints they create.

```bash
# After approval: read-only checks can list cloud devices when a key is present.
pytest --smoke tests/unit/smoke/test_00_requirements.py -s

# Only after approval and reviewing prerequisites and cleanup:
pytest --smoke tests/unit/smoke
```

Before submitting, follow [CONTRIBUTING.md](CONTRIBUTING.md): docstrings,
security checks (`bandit -r mobilerun`, `safety scan`) and external docs updates.

## Code rules

- Feature packages mirror the top-level layout with few files and no extra
  subfolders beyond `cli/`.
- Keep config/settings modules declarative; put logic in role-named modules and
  helpers in `utils.py`.
- Put tunable constants (URLs, timeouts, names) in the package's `settings.py`,
  built like `mobilerun/config/settings.py`.
- Ship each feature or fix with unit tests using fakes, without devices or cost.
- Do not add speculative functions: add code used by a caller or public API.
- Use type hints and explicit contracts. Device capabilities that are unavailable
  raise `DeviceOperationNotSupported`; do not guess capabilities on our classes.
- In device action flows, use `await device.wait(seconds)` rather than
  `asyncio.sleep`: recording captures device waits for macro replay.
- Use `async with` or `close()` for lifecycle cleanup. Release resources created
  in this run, not resources supplied by ID; preserve ownership on reuse.
- Billing defaults to `paid`; `MOBILERUN_BILLING` changes the default and an
  explicit `billing=` argument overrides both.
  Respect guards in `mobilerun/clients/cloud/billing.py`; `no_commerce` blocks
  purchases.
- Never expose secret values in logs, exceptions or representations. Cloud tasks
  use vault references from `CloudCredentials`, not local credential values.
- CLI commands should prompt for missing values when interactive, confirm paid
  or destructive actions, and support `--yes` and non-interactive operation.
- No compatibility layer: use the current public API and CLI command paths; agent code comes from the `mobilerun-agents` add-on.

## Documentation

- Give each package folder a `README.md`: purpose, quick example, files table and
  links to related documentation.
- Update the package README when changing a public API. Mark examples that can
  create paid resources or consume LLM tokens.

## Safety

- Never run paid operations, create cloud resources or make LLM calls without
  the user's approval. Check cleanup before running approved smoke tests.
- Never commit secrets. Cloud authentication can use `MOBILERUN_API_KEY` in the
  environment; never print its value.
