"""Config loading with platform-aware user config and migrations."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from mobilerun.config.settings import get_settings
from mobilerun.errors.base import ConfigError

from .migrations import CURRENT_VERSION, migrate
from .user_config import MobileConfig


class OutdatedConfigError(ConfigError):
    """Raised when user config is unreadable or invalid."""

    pass


class ConfigLoader:
    """Unified config loading with user config support."""

    APP_NAME = "droidrun"
    CONFIG_FILE = "config.yaml"
    config_type = MobileConfig

    @classmethod
    def get_user_config_dir(cls) -> Path:
        return get_settings().MOBILERUN_USER_CONFIG_DIR

    @classmethod
    def get_user_config_path(cls) -> Path:
        return cls.get_user_config_dir() / cls.CONFIG_FILE

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> MobileConfig:
        """
        Load config with resolution order:
        1. Explicit config_path argument
        2. MOBILERUN_CONFIG env var
        3. User config (platformdirs config dir for "droidrun", e.g.
           ~/Library/Application Support/droidrun/config.yaml on macOS,
           ~/.config/droidrun/config.yaml on Linux)
        4. Package defaults (creates user config)
        """
        if config_path:
            return cls._load_user_config(Path(config_path))

        env_config = get_settings().MOBILERUN_CONFIG
        if env_config and Path(env_config).exists():
            return cls._load_user_config(Path(env_config))

        user_config_path = cls.get_user_config_path()

        if user_config_path.exists():
            return cls._load_user_config(user_config_path)

        return cls._init_user_config()

    @classmethod
    def _load_user_config(cls, user_config_path: Path) -> MobileConfig:
        """Load user config and run migrations."""
        try:
            with open(user_config_path, "r", encoding="utf-8") as f:
                user_dict = yaml.safe_load(f)
            if not isinstance(user_dict, dict):
                raise ValueError("config must be a mapping")
            old_version = user_dict.get("_version", 1)
            if type(old_version) is not int or old_version < 1:
                raise ValueError("_version must be a positive integer")
            user_dict["_version"] = old_version
            user_dict = migrate(user_dict)
            config = cls.config_type.from_dict(user_dict)
        except (
            OSError,
            UnicodeError,
            yaml.YAMLError,
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            raise OutdatedConfigError(
                f"Invalid config at {user_config_path}: {exc}"
            ) from exc

        if user_dict.get("_version", 0) > old_version:
            backup = user_config_path.with_name(
                f"{user_config_path.name}.v{old_version}.bak"
            )
            fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as target, open(user_config_path, "rb") as source:
                shutil.copyfileobj(source, target)
            cls._save_dict(user_dict, user_config_path)

        return config

    @classmethod
    def _init_user_config(cls) -> MobileConfig:
        """Create user config from defaults on first run."""
        config = cls.config_type()
        cls.save(config)
        return config

    @classmethod
    def save(cls, config: MobileConfig) -> Path:
        """Save config to user config path."""
        config_dict = config.to_dict()
        config_dict["_version"] = CURRENT_VERSION
        return cls._save_dict(config_dict, cls.get_user_config_path())

    @classmethod
    def _save_dict(cls, config_dict: Dict[str, Any], path: Path) -> Path:
        """Save config dict to path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            if os.name != "nt":
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = -1  # fdopen owns the descriptor from here.
                f.seek(0)
                f.truncate()
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        finally:
            if fd != -1:
                os.close(fd)
        return path
