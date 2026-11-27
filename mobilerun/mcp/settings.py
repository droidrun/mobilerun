"""Environment-backed MCP settings."""

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPFormat(BaseModel):
    """Agent-specific MCP config format and authentication."""

    servers_key: str
    entry_type: str | None = None
    api_key_ref: str | None = None
    bearer_token_env_var: bool = False
    user_api_key: bool = True
    nested_servers: bool = False
    disable_oauth_with_api_key: bool = False
    jsonc_alternative: bool = False
    oauth_sign_in: str | None = None


DEFAULT_MCP_FORMATS = {
    "claude-code": MCPFormat(
        servers_key="mcpServers",
        entry_type="http",
        api_key_ref="${MOBILERUN_API_KEY}",
        user_api_key=False,
        oauth_sign_in="Next: sign in from Claude Code: run /mcp, select {server_name}, then Authenticate",
    ),
    "codex": MCPFormat(
        servers_key="mcp_servers",
        bearer_token_env_var=True,
        oauth_sign_in="Next: sign in with `codex mcp login {server_name}`",
    ),
    "opencode": MCPFormat(
        servers_key="mcp",
        entry_type="remote",
        api_key_ref="{env:MOBILERUN_API_KEY}",
        nested_servers=True,
        disable_oauth_with_api_key=True,
        jsonc_alternative=True,
        oauth_sign_in="Next: sign in with `opencode mcp auth {server_name}`",
    ),
    "cursor": MCPFormat(
        servers_key="mcpServers",
        api_key_ref="${env:MOBILERUN_API_KEY}",
        oauth_sign_in="Next: sign in from Cursor: open its MCP server settings, select {server_name}, then sign in",
    ),
}


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
