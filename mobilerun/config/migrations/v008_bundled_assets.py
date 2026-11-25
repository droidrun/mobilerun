"""Migration v8: relocate references to bundled runtime assets.

When the add-on is absent, leave legacy paths unchanged rather than guessing
which paths are bundled. The add-on's PathResolver falls back from legacy
``config/prompts`` and ``config/app_cards`` paths to its own assets when it is
installed later, even if this migration has already advanced the version.
"""

from importlib.util import find_spec
from pathlib import Path
from typing import Any

VERSION = 8
# CWD is a property of the reader, not of the shared config file.
READ_TIME_ONLY = True


def bundled_asset(path: Path) -> Path | None:
    """Find a read-only fallback for a legacy asset path, never importing agents."""
    if (
        len(path.parts) < 2
        or path.parts[0] != "config"
        or path.parts[1] not in {"prompts", "app_cards"}
        or ".." in path.parts
    ):
        return None
    package = Path(__file__).resolve().parents[2]
    try:
        # Locate add-on assets for old config paths without importing the add-on.
        spec = find_spec("mobilerun.agents")
    except ModuleNotFoundError as error:
        if error.name != "mobilerun.agents" and not (error.name or "").startswith(
            "mobilerun.agents."
        ):
            raise
        spec = None
    locations = spec.submodule_search_locations or () if spec else ()
    roots = [package, package / "agents", *(Path(p) for p in locations)]
    for root in roots:
        candidate = root.joinpath("assets", *path.parts[1:])
        if candidate.exists():
            return candidate
    return None


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
            if not path.exists() and bundled_asset(path) is not None:
                section[key] = str(Path("assets").joinpath(*path.parts[1:]))
    return config
