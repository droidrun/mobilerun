"""Shared protocol constants, runtime tunables and credential paths."""

from pathlib import Path
from typing import Literal

import platformdirs
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mobilerun.models.billing import Billing

_APP_NAME = "droidrun"  # Preserve the existing user configuration directory.


def _user_config_dir() -> Path:
    return Path(platformdirs.user_config_dir(_APP_NAME))


def _smoke_apk() -> str:
    return str(Path(platformdirs.user_cache_dir(_APP_NAME)) / "smoke" / "counter.apk")


class AgentPaths(BaseModel):
    """User and project paths for each agent integration."""

    skills: tuple[str, str | None] | None
    mcp: tuple[str, str | None] | None
    # Kept as untyped feature data so shared settings never import MCP.
    mcp_format: dict[str, object] | None = None


def _agents() -> dict[str, AgentPaths]:
    return {
        "claude-code": AgentPaths(
            skills=(".claude/skills", ".claude/skills"),
            mcp=(".claude.json", ".mcp.json"),
        ),
        "codex": AgentPaths(
            skills=(".agents/skills", ".agents/skills"),
            mcp=(".codex/config.toml", ".codex/config.toml"),
        ),
        "opencode": AgentPaths(
            skills=(".config/opencode/skills", ".opencode/skills"),
            mcp=(".config/opencode/opencode.json", "opencode.json"),
        ),
        "cursor": AgentPaths(
            skills=(".cursor/skills", ".cursor/skills"),
            mcp=(".cursor/mcp.json", ".cursor/mcp.json"),
        ),
    }


class Settings(BaseSettings):
    """Shared runtime settings, read from the environment on every get_settings() call."""

    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
        env_file=None,
        secrets_dir=None,
        use_attribute_docstrings=True,
    )

    # ── Paths / user config

    MOBILERUN_USER_CONFIG_DIR: Path = Field(default_factory=_user_config_dir)
    """User configuration directory."""

    MOBILERUN_CONFIG: str | None = None
    """User configuration file override."""

    MOBILERUN_CREDENTIALS_FILE: Path | None = None
    """Local credentials YAML (package → name → fields). Defaults to <MOBILERUN_USER_CONFIG_DIR>/credentials.yaml."""

    MOBILERUN_LOG_LEVEL: Literal["quiet", "normal", "verbose", "debug"] | None = None
    """Log level."""

    MOBILERUN_PORTAL_REMOTE_PORT: int = Field(8080, ge=1, le=65535)
    """Portal remote port."""
    MOBILERUN_AGENTS: dict[str, AgentPaths] = Field(default_factory=_agents)
    """Skill and MCP configuration paths by agent."""

    ANDROID_HOME: str | None = None
    """Android SDK location."""
    ANDROID_SDK_ROOT: str | None = None
    """Android SDK root directory."""
    MOBILERUN_SMOKE_APK: str = Field(default_factory=_smoke_apk)
    """Demo APK to install. Defaults to smoke/counter.apk in the user cache."""

    # ── Devices

    MOBILERUN_ANDROID_DEVICE_TYPE: str | None = None
    """Android device type."""

    MOBILERUN_CLOUD_DEVICE_TYPE: str = "android_cloud_phone"
    """Cloud device type."""

    # ── Cloud API and tasks

    MOBILERUN_BILLING: Billing = Billing.PAID
    """Charge ceiling: no_commerce, included or paid."""

    MOBILERUN_CLOUD_PROXY_COUNTRY: str = Field("us", min_length=1)
    """Cloud proxy country."""
    MOBILERUN_CLOUD_PROXY_SMART_IP: bool = True
    """Cloud proxy smart ip."""

    MOBILERUN_CLOUD_BASE_URL: str = "https://api.mobilerun.ai/v1"
    """Cloud base url."""

    MOBILERUN_CLOUD_API_TIMEOUT: float = 10.0
    """Cloud api timeout."""

    MOBILERUN_CLOUD_NUMBER_CAPACITY_CONCURRENCY: int = Field(10, ge=1)
    """Maximum concurrent country capacity lookups for the CLI."""

    MOBILERUN_CLOUD_NUMBER_READY_TIMEOUT_SECONDS: float = Field(600.0, gt=0)
    """Max seconds to wait for a new cloud number to become active."""

    MOBILERUN_CLOUD_APP_INSTALL_TIMEOUT_SECONDS: float = Field(180.0, gt=0)
    """Cloud app install timeout seconds."""

    MOBILERUN_CLOUD_TASK_TIMEOUT_SECONDS: float = 1800.0
    """Cloud task timeout seconds."""

    MOBILERUN_CLOUD_TASK_POLL_INTERVAL_SECONDS: float = 2.0
    """Cloud task poll interval seconds."""

    MOBILERUN_WEBHOOK_PAGE_SIZE: int = Field(100, ge=1, le=100)
    """Number of webhook endpoints requested per page."""

    MOBILERUN_WEBHOOK_PING_TIMEOUT: float = Field(15.0, gt=0, le=60)
    """Max seconds to wait for a signed webhook test delivery."""

    # ── Device authentication

    MOBILERUN_AUTH_URL: str = "https://cloud.mobilerun.ai/api/auth"
    """Auth url."""

    MOBILERUN_DEVICE_AUTH_REQUEST_TIMEOUT: float = 10.0
    """Device auth request timeout."""

    MOBILERUN_DEVICE_AUTH_MIN_POLL_INTERVAL: float = 5.0
    """Device auth min poll interval."""

    MOBILERUN_DEVICE_AUTH_SLOW_DOWN_BUMP: float = 5.0
    """Device auth slow down bump."""

    # ── UI state and retries

    MOBILERUN_STATE_RETRY_DELAYS: list[float] = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    """State retry delays."""

    MOBILERUN_STATE_MAX_RETRIES: int = 7
    """State max retries."""

    MOBILERUN_STATE_RECOVERY_AFTER_ATTEMPT: int = 5
    """State recovery after attempt."""

    MOBILERUN_STATE_MAX_TIMEOUT_ATTEMPTS: int = 2
    """State max timeout attempts."""

    MOBILERUN_STATE_SLOW_FETCH_NOTICE_S: float = 3.0
    """State slow fetch notice s."""

    # ── Model screenshots

    MOBILERUN_MODEL_SCREENSHOT_MAX_SIDE: int = 2048
    """Model screenshot max side."""

    @field_validator("MOBILERUN_AUTH_URL", "MOBILERUN_CLOUD_BASE_URL", mode="before")
    @classmethod
    def _empty_url_uses_default(cls, value: str, info):
        if value == "":
            return cls.model_fields[info.field_name].default
        return value

    @field_validator("MOBILERUN_CREDENTIALS_FILE", mode="before")
    @classmethod
    def _empty_credentials_uses_default(cls, value):
        return None if value == "" else value

    @property
    def local_credentials_path(self) -> Path:
        return (
            self.MOBILERUN_CREDENTIALS_FILE
            or self.MOBILERUN_USER_CONFIG_DIR / "credentials.yaml"
        ).expanduser()

    @property
    def mobilerun_cloud_credential_path(self) -> Path:
        return self.MOBILERUN_USER_CONFIG_DIR / "credentials" / "mobilerun-cloud.json"


def get_settings() -> Settings:
    return Settings()
