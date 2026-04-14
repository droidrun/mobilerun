"""
Prompt loading utility using Jinja2 templates.

Features:
- Loads from absolute file paths (resolved by AgentConfig + PathResolver)
- Conditional rendering: {% if variable %}...{% endif %}
- Loops with slicing: {% for item in items[-5:] %}...{% endfor %}
- Filters: {{ variable|default("fallback") }}
- Missing variables: silently ignored (renders as empty string)
- Extra variables: silently ignored
"""

from pathlib import Path
from typing import Any, Dict

import aiofiles
from jinja2 import Environment


class PromptLoader:
    """Jinja2 template renderer - loads from absolute file paths using aiofiles."""

    _env = None  # Cached Jinja2 environment

    @classmethod
    def _get_environment(cls) -> Environment:
        """Get or create cached Jinja2 environment."""
        if cls._env is None:
            cls._env = Environment(
                trim_blocks=True,  # Remove first newline after block
                lstrip_blocks=True,  # Strip leading whitespace before blocks
                keep_trailing_newline=False,
            )

        return cls._env

    @staticmethod
    async def load_prompt(file_path: str, variables: Dict[str, Any] = None) -> str:
        """
        Load and render Jinja2 template from absolute file path.

        Path resolution is handled by AgentConfig + PathResolver.
        This method just loads and renders.

        Args:
            file_path: ABSOLUTE path to template file (from AgentConfig methods)
            variables: Dict of variables to pass to template
                      - Missing variables: silently ignored (render as empty string)
                      - Extra variables: silently ignored

        Returns:
            Rendered prompt string

        Raises:
            FileNotFoundError: If template file doesn't exist

        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        # Read template content
        async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
            template_content = await f.read()

        # Use render_template for actual rendering
        return PromptLoader.render_template(template_content, variables)

    @staticmethod
    def render_template(template_string: str, variables: Dict[str, Any] = None) -> str:
        """
        Render Jinja2 template from string (NOT file path).

        This is used for custom prompts passed at runtime.

        Args:
            template_string: Jinja2 template as string
            variables: Dict of variables to pass to template
                      - Missing variables: silently ignored (render as empty string)
                      - Extra variables: silently ignored

        Returns:
            Rendered prompt string

        """
        # Get cached environment and create template from string
        env = PromptLoader._get_environment()
        template = env.from_string(template_string)

        # Render with variables (empty dict if None)
        # Missing variables render as empty string (default Undefined behavior)
        # Extra variables are silently ignored
        return template.render(**(variables or {}))
