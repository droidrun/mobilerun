"""Migration v8: surface workflow timeout through AgentConfig.

Before v8 the CLI hard-coded ``MobileAgent(timeout=1000)`` and there was no way
to change it from configuration. v8 promotes the timeout to an ``AgentConfig``
field so it can be tuned via ``mobilerun configure → Advanced settings`` (or
directly in YAML). Old configs need the default value written into the agent
section so they round-trip identically after migration.
"""

from typing import Any, Dict

VERSION = 8

_DEFAULT_TIMEOUT_SECONDS = 1000


def migrate(config: Dict[str, Any]) -> Dict[str, Any]:
    agent = config.get("agent")
    if not isinstance(agent, dict):
        agent = {}
        config["agent"] = agent
    agent.setdefault("timeout", _DEFAULT_TIMEOUT_SECONDS)
    return config
