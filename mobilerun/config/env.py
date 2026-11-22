"""Typed, masked environment credentials, read afresh on every access."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
        env_file=None,
        secrets_dir=None,
        use_attribute_docstrings=True,
    )
    MOBILERUN_API_KEY: SecretStr | None = None
    """Mobilerun api key."""
    MOBILERUN_DEVICE_TOKEN: SecretStr | None = None
    """Mobilerun device token."""


def get_secrets() -> Secrets:
    return Secrets()


def secret_value(value: SecretStr | None) -> str | None:
    """Unmask a typed secret only at the point where a consumer needs it."""
    return value.get_secret_value() if value is not None else None
