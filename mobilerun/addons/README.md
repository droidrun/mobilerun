# addons/

Discovery and actionable errors for optional add-on packages. Discovery does
not import the add-on itself.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Package marker. |
| `agents.py` | Detect the optional agents add-on and report missing installs. |

## Quick example

```python
from mobilerun.addons.agents import agents_available

print(agents_available())
```

See the [package map](../README.md) and [CLI](../cli/README.md).
