# clients/

**Purpose:** Functional SDK and Portal transport boundaries.

**What belongs here / what doesn't:** Put service calls and response handling here, not CLI printing or core, agent, and feature imports.

**Contents:**

- `__init__.py` — Package marker.
- `cloud/` — Cloud authentication and account access.
- `local/` — Function-only local helpers, APK metadata and shared code waiting.
- `portal/` — Portal setup, HTTP, and state retrieval.

**Examples:**

Use [devices](../core/devices/README.md) for the public API. Cloud, local and Portal clients are lower-level transport helpers.

**Depends on / used by:** Uses shared configuration, errors, and external clients; used by core, CLI, and features.

See [cloud](cloud/README.md), [local](local/README.md), and [Portal](portal/README.md).
