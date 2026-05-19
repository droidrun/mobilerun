import logging
import os
from typing import Any, Dict, Optional

import yaml

from mobilerun.config_manager.path_resolver import PathResolver
from mobilerun.credential_manager.credential_manager import (
    CredentialManager,
    CredentialNotFoundError,
)

logger = logging.getLogger("mobilerun")


class FileCredentialManager(CredentialManager):
    """
    Credential manager that supports both dict and YAML file sources.

    Secret values can be provided directly, read from environment variables, or
    read from local secret files.
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
        from mobilerun.config_manager.config_manager import CredentialsConfig

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
        for secret_id, secret_data in credentials_dict.items():
            value = self._resolve_secret_data(secret_id, secret_data)
            if value:
                secrets[secret_id] = value
        return secrets

    def _load_from_file(self, file_path: str) -> Dict[str, str]:
        """
        Load credentials from YAML file.

        File format:
            secrets:
              MY_PASSWORD:
                value: "secret123"
                enabled: true
              USERNAME:
                env: APP_USERNAME
              TOKEN:
                file: ~/.config/my-app/token
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
            value = self._resolve_secret_data(secret_id, secret_data)
            if value:
                secrets[secret_id] = value
                logger.debug(f"Loaded secret: {secret_id}")

        return secrets

    def _resolve_secret_data(self, secret_id: str, secret_data: Any) -> Optional[str]:
        """Resolve a secret entry without logging the resolved value."""
        if isinstance(secret_data, str):
            return secret_data or None

        if not isinstance(secret_data, dict):
            logger.warning(
                f"Skipped invalid secret: {secret_id} (type={type(secret_data)})"
            )
            return None

        enabled = secret_data.get("enabled", True)
        if not enabled:
            logger.debug(f"Skipped disabled secret: {secret_id}")
            return None

        if "value" in secret_data:
            value = secret_data.get("value", "")
            if isinstance(value, str) and value:
                return value
            logger.warning(f"Skipped secret '{secret_id}': empty or invalid value")
            return None

        if "env" in secret_data:
            env_name = secret_data.get("env")
            if not isinstance(env_name, str) or not env_name:
                logger.warning(f"Skipped secret '{secret_id}': invalid env source")
                return None
            value = os.environ.get(env_name, "")
            if value:
                return value
            logger.warning(
                f"Skipped secret '{secret_id}': environment variable '{env_name}' is not set"
            )
            return None

        if "file" in secret_data:
            source_path = secret_data.get("file")
            if not isinstance(source_path, str) or not source_path:
                logger.warning(f"Skipped secret '{secret_id}': invalid file source")
                return None
            try:
                resolved_path = PathResolver.resolve(source_path, must_exist=True)
                with open(resolved_path, "r") as f:
                    value = f.read().strip()
            except Exception as exc:
                logger.warning(
                    f"Skipped secret '{secret_id}': could not read file source ({exc})"
                )
                return None
            if value:
                return value
            logger.warning(f"Skipped secret '{secret_id}': file source is empty")
            return None

        logger.warning(
            f"Skipped secret '{secret_id}': expected one of value, env, or file"
        )
        return None

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
