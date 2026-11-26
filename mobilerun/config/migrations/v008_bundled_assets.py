"""Migration v8: relocate references to prompts bundled in the base package.

Leave add-on-only prompts under ``config/prompts`` so PathResolver can still
find them in the add-on.
"""

from pathlib import Path
from typing import Any

from mobilerun.utils.paths import PathResolver

VERSION = 8
# CWD is a property of the reader, not of the shared config file.
READ_TIME_ONLY = True


def migrate(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve legacy prompt paths for an in-memory config only."""
    agent = config.get("agent")
    if not isinstance(agent, dict):
        return config
    for section in agent.values():
        if not isinstance(section, dict):
            continue
        for key in ("system_prompt", "user_prompt"):
            value = section.get(key)
            if not isinstance(value, str):
                continue
            path = Path(value)
            if path.parts[:2] != ("config", "prompts") or ".." in path.parts:
                continue
            asset = Path("assets", *path.parts[1:])
            if (PathResolver.get_project_root() / asset).exists() and (
                PathResolver.resolve(path) == PathResolver.resolve(asset)
            ):
                section[key] = str(asset)
    return config
