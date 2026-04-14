"""External agent loader — dynamic imports.

External agents are self-contained modules that receive raw ADB access
via ``async_adbutils.AdbDevice``. They bring their own LLM client, prompts,
parsing, and action loop — zero imports from ``droidrun``.

An external agent can be either:
- A single file: ``droidrun/agent/external/my_agent.py``
- A package:     ``droidrun/agent/external/my_agent/__init__.py``

Required contract::

    from async_adbutils import AdbDevice

    async def run(
        device: AdbDevice,       # raw ADB, already connected
        instruction: str,        # the task
        config: dict,            # from external_agents.<name> in config.yaml
        max_steps: int,          # step limit
    ) -> dict:                   # {"success": bool, "reason": str, "steps": int}

Optional: ``DEFAULT_CONFIG: dict`` — merged under the user's config.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

logger = logging.getLogger("droidrun")

_EXTERNAL_DIR = Path(__file__).parent


class ExternalAgentModule(TypedDict):
    """Type for a loaded external agent module."""

    run: Callable
    config: Dict[str, Any]


def list_agents() -> List[str]:
    """Discover available external agents by scanning the external/ directory.

    Returns:
        Sorted list of agent names (module stems or package directory names).
    """
    agents: list[str] = []
    for item in _EXTERNAL_DIR.iterdir():
        if item.name.startswith(("_", ".")):
            continue
        if item.is_file() and item.suffix == ".py":
            agents.append(item.stem)
        elif item.is_dir() and (item / "__init__.py").exists():
            agents.append(item.name)
    return sorted(agents)


def load_agent(name: str) -> Optional[ExternalAgentModule]:
    """Dynamically load an external agent by name.

    Args:
        name: Agent module name (e.g., ``"my_agent"``).

    Returns:
        Dict with ``run`` function and ``config`` defaults, or *None* on failure.
    """
    try:
        module = importlib.import_module(f"droidrun.agent.external.{name}")

        if not hasattr(module, "run"):
            logger.error(f"External agent '{name}' missing run() function")
            return None

        if not asyncio.iscoroutinefunction(module.run):
            logger.error(
                f"External agent '{name}' has a sync run() — it must be async (async def run)"
            )
            return None

        return {
            "run": module.run,
            "config": getattr(module, "DEFAULT_CONFIG", {}),
        }

    except ImportError as e:
        logger.error(f"Failed to load external agent '{name}': {e}")
        return None
