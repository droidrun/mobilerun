"""Migration v4: Remove deprecated agent configs and rename prompt paths."""

from typing import Any, Dict

VERSION = 4

_OLD_SYSTEM_PROMPT = "config/prompts/codeact/tools_system.jinja2"
_OLD_USER_PROMPT = "config/prompts/codeact/tools_user.jinja2"
_NEW_SYSTEM_PROMPT = "config/prompts/fast_agent/system.jinja2"
_NEW_USER_PROMPT = "config/prompts/fast_agent/user.jinja2"


def migrate(config: Dict[str, Any]) -> Dict[str, Any]:
    """Strip removed agent configs and update prompt paths."""
    agent = config.get("agent", {})

    fast_agent = agent.get("fast_agent", {})

    # Remove deprecated fields from fast_agent
    fast_agent.pop("codeact", None)
    fast_agent.pop("safe_execution", None)
    fast_agent.pop("execution_timeout", None)

    # Update prompt paths from legacy codeact directory
    if fast_agent.get("system_prompt") == _OLD_SYSTEM_PROMPT:
        fast_agent["system_prompt"] = _NEW_SYSTEM_PROMPT
    if fast_agent.get("user_prompt") == _OLD_USER_PROMPT:
        fast_agent["user_prompt"] = _NEW_USER_PROMPT

    # Remove scripter section
    agent.pop("scripter", None)

    # Remove top-level safe_execution
    config.pop("safe_execution", None)

    # Remove deprecated LLM profiles
    profiles = config.get("llm_profiles", {})
    profiles.pop("text_manipulator", None)
    profiles.pop("scripter", None)

    return config
