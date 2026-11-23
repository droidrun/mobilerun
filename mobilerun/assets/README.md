# assets/

Bundled prompt templates and state guidance used by the framework.

**What belongs here / what doesn't:** Put runtime assets here, not Python implementations or generated files.

## Contents

| Path | Purpose |
| --- | --- |
| `prompts/` | Prompt templates and state guidance |

## Quick example

```python
from mobilerun.utils.prompts import load_prompt_text

guidance = load_prompt_text("state/screenshot_only_guidance.md")
```

See [prompt utilities](../utils/README.md).

**Depends on / used by:** Read through shared prompt utilities; used by devices and state-related features.
