"""Environment-backed defaults for macro replay."""

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    MOBILERUN_MACRO_DIRECTORY: str = "trajectories"
    """Default directory for the macro list command."""


def get_macro_settings() -> MacroSettings:
    """Read current macro settings from the environment."""
    return MacroSettings()
