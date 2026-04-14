import logging
from typing import Any, Dict, Optional

import yaml

from droidrun.config_manager.path_resolver import PathResolver
from droidrun.credential_manager.credential_manager import (
    CredentialManager,
    CredentialNotFoundError,
)

logger = logging.getLogger("droidrun")


class FileCredentialManager(CredentialManager):
    """
    Credential manager that supports both dict and YAML file sources.
    """

    def __init__(self, credentials: Any):
        """
        Initialize credential manager from dict or file path.

        Args:
            credentials: Either dict or string (file path) or CredentialsConfig
        """
        self.path: Optional[str] = None
        self.secrets = self._load(credentials)

        if self.path:
            logger.debug(f"✅ Loaded {len(self.secrets)} secrets from {self.path}")
        else:
            logger.debug(f"✅ Loaded {len(self.secrets)} secrets from in-memory dict")

    def _load(self, credentials: Any) -> Dict[str, str]:
        """Load credentials from dict or file."""
        from droidrun.config_manager.config_manager import CredentialsConfig

        # Dict mode
        if isinstance(credentials, dict):
            return self._load_from_dict(credentials)

        # CredentialsConfig mode
        if isinstance(credentials, CredentialsConfig):
            if not credentials.enabled:
                logger.debug("Credentials disabled in config")
                return {}
            self.path = credentials.file_path
            return self._load_from_file(credentials.file_path)

        # String mode (direct file path)
        if isinstance(credentials, str):
            self.path = credentials
            return self._load_from_file(credentials)

        logger.warning(f"Unknown credentials type: {type(credentials)}")
        return {}

    def _load_from_dict(self, credentials_dict: dict) -> Dict[str, str]:
        """Load credentials from in-memory dict."""
        secrets = {}
        for secret_id, secret_value in credentials_dict.items():
            if isinstance(secret_value, str) and secret_value:
                secrets[secret_id] = secret_value
            else:
                logger.warning(
                    f"Skipped invalid secret: {secret_id} (type={type(secret_value)})"
                )
        return secrets

    def _load_from_file(self, file_path: str) -> Dict[str, str]:
        """
        Load credentials from YAML file.

        File format:
            secrets:
              MY_PASSWORD:
                value: "secret123"
                enabled: true
              SIMPLE_KEY: "simple_value"  # Auto-enabled

        Returns:
            Dict of enabled secrets {secret_id: secret_value}
        """
        path = PathResolver.resolve(file_path, must_exist=True)
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not data or "secrets" not in data:
            logger.warning(f"No 'secrets' section found in {path}")
            return {}

        secrets = {}
        for secret_id, secret_data in data["secrets"].items():
            if isinstance(secret_data, dict):
                enabled = secret_data.get("enabled", True)
                value = secret_data.get("value", "")
            else:
                enabled = True
                value = secret_data

            if enabled and value:
                secrets[secret_id] = value
                logger.debug(f"Loaded secret: {secret_id}")
            else:
                logger.debug(
                    f"Skipped secret: {secret_id} (enabled={enabled}, has_value={bool(value)})"
                )

        return secrets

    async def resolve_key(self, key: str) -> str:
        """Get secret value by key."""
        logger.debug(f"🔑 Accessing secret: '{key}'")

        if key not in self.secrets:
            available = list(self.secrets.keys())
            raise CredentialNotFoundError(
                f"Secret '{key}' not found. Available: {available}"
            )

        return self.secrets[key]

    async def get_keys(self) -> list[str]:
        """Get all available credential keys."""
        return list(self.secrets.keys())

    def has_credential(self, secret_id: str) -> bool:
        """Check if secret ID exists."""
        return secret_id in self.secrets

    def __repr__(self) -> str:
        """String representation."""
        count = len(self.secrets)
        if self.path:
            return f"<FileCredentialManager path={self.path} secrets={count}>"
        return f"<FileCredentialManager mode=dict secrets={count}>"
