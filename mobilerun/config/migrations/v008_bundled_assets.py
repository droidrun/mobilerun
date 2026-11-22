"""Migration v8: relocate references to bundled runtime assets.

When the add-on is absent, leave legacy paths unchanged rather than guessing
which paths are bundled. The add-on's PathResolver falls back from legacy
``config/prompts`` and ``config/app_cards`` paths to its own assets when it is
installed later, even if this migration has already advanced the version.
"""

from pathlib import Path
from typing import Any

from mobilerun.addons.agents import agents_locations

VERSION = 8


def migrate(config: dict[str, Any]) -> dict[str, Any]:
    agent = config.get("agent")
    if not isinstance(agent, dict):
        return config
    assets = Path(__file__).resolve().parents[2] / "assets"
    addon_assets = [Path(location) / "assets" for location in agents_locations()]
    for section in agent.values():
        if not isinstance(section, dict):
            continue
        for key in ("system_prompt", "user_prompt", "app_cards_dir"):
            value = section.get(key)
            if not isinstance(value, str):
                continue
            path = Path(value)
            if path.is_absolute() or len(path.parts) < 2 or path.parts[0] != "config":
                continue
            if path.parts[1] not in {"prompts", "app_cards"} or ".." in path.parts:
                continue
            # A caller-provided CWD resource wins over the bundled fallback.
            if path.exists():
                continue
            target = assets.joinpath(*path.parts[1:])
            relocated_target = (
                assets.parent / "agents" / "assets" / Path(*path.parts[1:])
            )
            if (
                target.exists()
                or relocated_target.exists()
                or any(root.joinpath(*path.parts[1:]).exists() for root in addon_assets)
            ):
                section[key] = str(Path("assets").joinpath(*path.parts[1:]))
    return config
