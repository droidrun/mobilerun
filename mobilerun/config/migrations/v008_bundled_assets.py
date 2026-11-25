"""Migration v8: relocate references to bundled runtime assets.

When the add-on is absent, leave legacy paths unchanged rather than guessing
which paths are bundled. If the add-on is installed later, PathResolver can
still resolve legacy ``config/prompts`` and ``config/app_cards`` paths.
"""

from pathlib import Path
from typing import Any

from mobilerun.utils.paths import PathResolver

VERSION = 8
# CWD is a property of the reader, not of the shared config file.
READ_TIME_ONLY = True


def migrate(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve legacy paths for an in-memory config only."""
    agent = config.get("agent")
    if not isinstance(agent, dict):
        return config
    for section in agent.values():
        if not isinstance(section, dict):
            continue
        for key in ("system_prompt", "user_prompt", "app_cards_dir"):
            value = section.get(key)
            if not isinstance(value, str):
                continue
            path = Path(value)
            if not path.exists() and PathResolver.bundled_asset(path) is not None:
                section[key] = str(Path("assets").joinpath(*path.parts[1:]))
    return config
