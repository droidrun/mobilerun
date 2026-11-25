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
- LLM-backed agent execution is in the separate `mobilerun-agents` package (not yet publicly published).
- `mobilerun/clients/`: cloud, local and Portal service adapters.
- `mobilerun/models/`, `config/`, `errors/`, `logs/`, `utils/`: shared types and support.
- `mobilerun/cli/`: command groups and shared CLI helpers.
- `mobilerun/macro/`, `sandbox/`, `skills/`: independent framework features.
- `mobilerun-agents` (separate package): agent execution, integrations and agent commands.
- `tests/unit/`: unit tests and opt-in real-world smoke tests.
- `skills/`: git submodule; `hatch_build.py` bundles its files in the wheel.

Shared core must not import agents or feature packages. The intended boundaries
are checked in `tests/unit/test_import_boundaries.py`. The agent-required
dependency test (`tests/unit/test_agent_required_dependencies.py`) lives in the
separate `mobilerun-agents` repository. The CLI entry point has a special role; see the boundary tests for exceptions.
The v008 migration and legacy path resolver locate `mobilerun_agents` assets without importing the add-on.

## Where code goes

Shared `errors/`, `models/`, `utils/`, `cli/`, `config/`, and `logs/` serve
core and clients. Features (`macro/`, `mcp/`, `skills/`, `sandbox/`) keep their
own CLI, errors, models, settings, and utilities inside their folders. Features
may import shared code, core, and clients; core may import clients; clients
never import core or the root package; shared code never imports features.
Register feature CLI groups via `mobilerun.cli` entry points in `pyproject.toml`.

| New code | Location |
| --- | --- |
| Device error | `errors/device.py` |
| Macro error | `macro/errors.py` |
| Proxy CLI command | `cli/proxy/` |
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
pytest tests/unit/test_agent_config_boundary.py
pytest tests/unit/test_import_boundaries.py::test_shared_packages_do_not_import_features

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
