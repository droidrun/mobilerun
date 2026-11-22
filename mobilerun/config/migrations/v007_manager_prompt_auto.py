"""Migration v7: use automatic manager prompt selection by default.

Configs produced before v7 stored the bundled stateful manager prompt as an
explicit path. That pin prevents ``stateless: true`` from selecting the bundled
stateless prompt. Convert only that exact former default to ``None``. Any other
path remains an intentional override; an intentional pin to the former default
can be restored after migration.
"""

from typing import Any, Dict

VERSION = 7

_OLD_DEFAULT_MANAGER_SYSTEM_PROMPT = "config/prompts/manager/system.jinja2"


def migrate(config: Dict[str, Any]) -> Dict[str, Any]:
    agent = config.get("agent")
    if not isinstance(agent, dict):
        return config

    manager = agent.get("manager")
    if not isinstance(manager, dict):
        return config

    if manager.get("system_prompt") == _OLD_DEFAULT_MANAGER_SYSTEM_PROMPT:
        manager["system_prompt"] = None

    return config
