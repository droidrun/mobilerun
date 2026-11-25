# utils/

Core's utilities. Feature packages keep their own utilities inside their folder.

Standalone helpers for coordinates, images, verification codes and timestamps.

**What belongs here / what doesn't:** Helpers must not depend on Mobilerun domain packages, settings, devices or CLI behavior.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Package marker |
| `coordinates.py`, `geometry.py`, `images.py` | Coordinates, bounds and image helpers |
| `codes.py` | Verification code extraction and polling |
| `timestamps.py` | API timestamp formatting |
| `redaction.py` | `secret_scope` redacts sensitive log values; `mask_url` preserves HTTP(S) hosts and paths while dropping userinfo and fragments and masking query values. |
| `paths.py`, `prompts.py` | Resolve package assets and load prompts. Legacy `config/prompts` and `config/app_cards` paths can read `mobilerun_agents/assets` without importing the add-on. |

Device-facing helpers live in `clients/local/`.

## Quick example

```python
from mobilerun.utils.coordinates import to_absolute

x, y = to_absolute(500, 500, 1080, 1920)
print(x, y)  # 540 960
```

Cloud helpers live in [clients/cloud](../clients/cloud/README.md).

**Depends on / used by:** Uses standard and third-party libraries; used across the framework.
