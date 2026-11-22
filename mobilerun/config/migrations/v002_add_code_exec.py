"""Migration v2: Rename codeact config to fast_agent, add codeact flag."""

from typing import Any, Dict

VERSION = 2

_OLD_SYSTEM_PROMPT = "config/prompts/codeact/system.jinja2"
_OLD_USER_PROMPT = "config/prompts/codeact/user.jinja2"
_NEW_SYSTEM_PROMPT = "config/prompts/codeact/tools_system.jinja2"
_NEW_USER_PROMPT = "config/prompts/codeact/tools_user.jinja2"


def migrate(config: Dict[str, Any]) -> Dict[str, Any]:
    """Rename agent.codeact -> agent.fast_agent and llm_profiles.codeact -> fast_agent."""
    agent = config.get("agent", {})

    # Rename agent.codeact -> agent.fast_agent
    if "codeact" in agent and "fast_agent" not in agent:
        agent["fast_agent"] = agent.pop("codeact")

    fast_agent = agent.setdefault("fast_agent", {})

    fast_agent.setdefault("codeact", False)
    fast_agent.setdefault("parallel_tools", True)

    # Remove safe_execution and execution_timeout (now in top-level safe_execution)
    fast_agent.pop("safe_execution", None)
    fast_agent.pop("execution_timeout", None)

    # Update prompt paths if still at old defaults
    if fast_agent.get("system_prompt") == _OLD_SYSTEM_PROMPT:
        fast_agent["system_prompt"] = _NEW_SYSTEM_PROMPT
    if fast_agent.get("user_prompt") == _OLD_USER_PROMPT:
        fast_agent["user_prompt"] = _NEW_USER_PROMPT

    config["agent"] = agent

    # Rename llm_profiles.codeact -> llm_profiles.fast_agent
    profiles = config.get("llm_profiles", {})
    if "codeact" in profiles and "fast_agent" not in profiles:
        profiles["fast_agent"] = profiles.pop("codeact")

    return config
