"""Environment-backed MCP settings."""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPSettings(BaseSettings):
    """MCP settings read from the environment on each call."""

    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
        env_file=None,
        secrets_dir=None,
        use_attribute_docstrings=True,
    )

    MOBILERUN_MCP_URL: str = "https://cloud.mobilerun.ai/api/mcp"
    """URL of the Mobilerun MCP server."""

    MOBILERUN_MCP_SERVER_NAME: str = "mobilerun"
    """Server key used in coding-agent configurations."""

    @field_validator("MOBILERUN_MCP_URL", mode="before")
    @classmethod
    def _empty_url_uses_default(cls, value: str, info):
        if value == "":
            return cls.model_fields[info.field_name].default
        return value


def get_mcp_settings() -> MCPSettings:
    """Read a fresh MCP settings snapshot."""
    return MCPSettings()
