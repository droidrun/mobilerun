"""Settings for sandbox authoring and local storage."""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mobilerun.config.settings import get_settings

RUNTIME_MODULE = "mobilerun_functions"
WEBHOOK_STEP_ID = "webhook"


class SandboxSettings(BaseSettings):
    """Sandbox runtime and deployment defaults."""

    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
        env_file=None,
        secrets_dir=None,
        use_attribute_docstrings=True,
    )

    MOBILERUN_SANDBOX_INSPECTION_TIMEOUT: float = 120.0
    MOBILERUN_SANDBOX_POLL_INTERVAL: float = 2.0
    MOBILERUN_SANDBOX_RUN_TIMEOUT: float = 300.0
    MOBILERUN_SANDBOX_FUNCTION_TIMEOUT: int = 600
    MOBILERUN_SANDBOX_FLOW_SUFFIX: str = " · flow"
    MOBILERUN_SANDBOX_ACTION_SUFFIX: str = " · function"
    MOBILERUN_SANDBOX_WEBHOOK_SUFFIX: str = " · webhook"
    MOBILERUN_SANDBOX_DIRECTORY: str | None = None

    @field_validator("MOBILERUN_SANDBOX_DIRECTORY", mode="before")
    @classmethod
    def _empty_directory_uses_default(cls, value: str | None) -> str | None:
        return None if value == "" else value

    @property
    def sandbox_directory(self) -> str:
        """Return the override or the user configuration directory."""
        return self.MOBILERUN_SANDBOX_DIRECTORY or str(
            get_settings().MOBILERUN_USER_CONFIG_DIR / "sandbox"
        )


def get_sandbox_settings() -> SandboxSettings:
    """Read current sandbox settings from the environment."""
    return SandboxSettings()
