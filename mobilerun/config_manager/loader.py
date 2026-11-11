"""Config loading with platform-aware user config and migrations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import platformdirs
import yaml

from .config_manager import MobileConfig
from .migrations import CURRENT_VERSION, migrate


class OutdatedConfigError(Exception):
    """Raised when user config is missing _version field."""

    pass


class ConfigLoader:
    """Unified config loading with user config support."""

    APP_NAME = "droidrun"
    CONFIG_FILE = "config.yaml"

    @classmethod
    def get_user_config_dir(cls) -> Path:
        return Path(platformdirs.user_config_dir(cls.APP_NAME))

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

        env_config = os.environ.get("MOBILERUN_CONFIG")
        if env_config and Path(env_config).exists():
            return cls._load_user_config(Path(env_config))

        user_config_path = cls.get_user_config_path()

        if user_config_path.exists():
            return cls._load_user_config(user_config_path)

        return cls._init_user_config()

    @classmethod
    def _load_user_config(cls, user_config_path: Path) -> MobileConfig:
        """Load user config and run migrations."""
        with open(user_config_path, "r", encoding="utf-8") as f:
            user_dict = yaml.safe_load(f) or {}

        if "_version" not in user_dict:
            raise OutdatedConfigError(
                f"Config at {user_config_path} is outdated (missing _version).\n"
                "Please update your config based on the latest example:\n"
                "https://github.com/droidrun/mobilerun/blob/main/mobilerun/config_example.yaml"
            )

        old_version = user_dict["_version"]
        user_dict = migrate(user_dict)

        if user_dict.get("_version", 0) > old_version:
            cls._save_dict(user_dict, user_config_path)

        return MobileConfig.from_dict(user_dict)

    @classmethod
    def _init_user_config(cls) -> MobileConfig:
        """Create user config from defaults on first run."""
        config = MobileConfig()
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
