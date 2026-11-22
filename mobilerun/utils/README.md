# utils/

Standalone helpers for coordinates, images, verification codes and timestamps.

**What belongs here / what doesn't:** Helpers must not depend on Mobilerun domain packages, settings, devices or CLI behavior.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Package marker |
| `coordinates.py`, `geometry.py`, `images.py` | Coordinates, bounds and image helpers |
| `codes.py` | Verification code extraction |
| `timestamps.py` | API timestamp formatting |

Device-facing helpers live in `clients/local/`.

## Quick example

```python
from mobilerun.utils.coordinates import to_absolute

x, y = to_absolute(500, 500, 1080, 1920)
print(x, y)  # 540 960
```

Package paths and prompt loading live in [config](../config/README.md); cloud helpers live in [clients/cloud](../clients/cloud/README.md).

**Depends on / used by:** Uses standard and third-party libraries; used across the framework.
