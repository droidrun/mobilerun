"""Environment-backed defaults for macro replay."""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mobilerun.config.settings import get_settings


class MacroSettings(BaseSettings):
    """Defaults used when callers do not supply replay options."""

    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
        env_file=None,
        secrets_dir=None,
        use_attribute_docstrings=True,
    )

    MOBILERUN_MACRO_DELAY: float = 1.0
    """Seconds between recorded actions."""

    MOBILERUN_MACRO_STATE_TIMEOUT: float = 5.0
    """Seconds allowed for a state guard to match."""

    MOBILERUN_MACRO_STATE_THRESHOLD: float = 0.85
    """Minimum state similarity for a guard to match."""

    MOBILERUN_MACRO_POLL_INTERVAL: float = 0.5
    """Seconds between state guard checks."""

    MOBILERUN_MACRO_DIRECTORY: str | None = None
    """Optional directory override for the macro list command."""

    @field_validator("MOBILERUN_MACRO_DIRECTORY", mode="before")
    @classmethod
    def _empty_macro_directory_uses_default(cls, value: str | None) -> str | None:
        return None if value == "" else value

    @property
    def macro_directory(self) -> str:
        """Return the configured directory or the user config default."""
        return self.MOBILERUN_MACRO_DIRECTORY or str(
            get_settings().MOBILERUN_USER_CONFIG_DIR / "macros"
        )


def get_macro_settings() -> MacroSettings:
    """Read current macro settings from the environment."""
    return MacroSettings()
