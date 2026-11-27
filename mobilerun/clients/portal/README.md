# portal/

**Purpose:** Portal companion-app setup and state access.

**What belongs here / what doesn't:** Put Portal service operations here, not CLI presentation or feature logic.

**Contents:**

- `__init__.py` — Public Portal types.
- `errors_text.py` — Portal error text constants.
- `http.py` — Portal HTTP URL and driver helpers.
- `install.py` — Portal installation and service readiness.
- `releases.py` — Latest Portal release lookup.
- `setup.py` — Portal setup argument helpers.
- `state_fetch.py` — State-fetch retries and driver recovery.

**Examples:**

```python
from mobilerun.clients.portal.http import is_portal_http_url

is_http = is_portal_http_url("http://localhost:8080")
```

**Depends on / used by:** Uses shared configuration, errors, and drivers; used by device setup and state providers.

## CLI device selection

DEVICE is optional: run `mobilerun device connect <id>` to save a default, or choose from discovered devices. An explicit DEVICE takes precedence, followed by the saved `device.serial` and discovery. Run `mobilerun device disconnect` to clear the default. Python device APIs remain explicit.

See [clients](../README.md) and [devices](../../core/devices/README.md).
