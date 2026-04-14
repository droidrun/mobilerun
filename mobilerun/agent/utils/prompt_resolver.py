"""
Prompt resolution utility for handling custom prompts passed as dict.

This module provides logic to resolve prompts from either:
1. Custom prompt strings (Jinja2 templates passed at runtime)
2. File paths (default behavior via AgentConfig)

This keeps the DroidAgent clean and separates prompt resolution logic.
"""

from typing import Dict, Optional


class PromptResolver:
    """
    Resolves prompts from custom dict or falls back to file paths.

    Usage:
        resolver = PromptResolver(custom_prompts={"fast_agent_system": "..."})
        prompt = resolver.get_prompt("fast_agent_system", fallback_path="/path/to/prompt.j2")
    """

    def __init__(self, custom_prompts: Optional[Dict[str, str]] = None):
        """
        Initialize prompt resolver.

        Args:
            custom_prompts: Dict mapping prompt keys to Jinja2 template strings.
                           Keys: "fast_agent_system", "fast_agent_user", "manager_system",
                                 "executor_system"
        """
        self.custom_prompts = custom_prompts or {}

    def get_prompt(
        self, prompt_key: str, fallback_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Get prompt by key, returning custom template or None if not found.

        Args:
            prompt_key: Prompt identifier (e.g., "fast_agent_system", "manager_system")
            fallback_path: Optional file path (unused, for API compatibility)

        Returns:
            Jinja2 template string if found in custom_prompts, else None
        """
        return self.custom_prompts.get(prompt_key)

    def has_custom_prompt(self, prompt_key: str) -> bool:
        """Check if a custom prompt exists for the given key."""
        return prompt_key in self.custom_prompts

    @staticmethod
    def get_valid_prompt_keys() -> list[str]:
        """
        Return list of valid prompt keys that can be customized.

        Returns:
            List of valid prompt key strings
        """
        return [
            "fast_agent_system",
            "fast_agent_user",
            "manager_system",
            "executor_system",
        ]
