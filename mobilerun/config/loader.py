"""Typed YAML configuration files and the shared Mobilerun configuration."""

from __future__ import annotations

import os
import stat
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, ClassVar, Generic, TypeVar

import yaml
from filelock import FileLock

from mobilerun.config.settings import get_settings
from mobilerun.errors.base import ConfigError

from .migrations import CURRENT_VERSION, migrate
from .user_config import MobileConfig


class OutdatedConfigError(ConfigError):
    """Raised when user config is unreadable or invalid."""

    pass


T = TypeVar("T")


class ConfigFile(ABC, Generic[T]):
    """Subclass to store a typed model in one YAML file in the user config directory."""

    CONFIG_FILE: ClassVar[str]
    config_type: type[T]
    WRITE_DEFAULTS: ClassVar[bool] = False
    USE_ENV_PATH: ClassVar[bool] = False
    CONFIG_VERSION: ClassVar[int | None] = None
    _invalid_config_error: ClassVar[type[ConfigError] | None] = None
    _override: ContextVar[T | None]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._override = ContextVar(f"{cls.__qualname__}.override", default=None)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> T:
        """Build and validate a model from the parsed mapping."""

    @classmethod
    @abstractmethod
    def to_dict(cls, config: T) -> dict[str, Any]:
        """Serialize a model to a YAML-ready mapping, validating it if needed."""

    @classmethod
    def parse_yaml(cls, text: str) -> object:
        """Parse YAML text, treating an empty document as an empty mapping."""
        data = yaml.safe_load(text)
        return {} if data is None else data

    @classmethod
    def migrate_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Upgrade a versioned mapping before validation and set its new _version."""
        return data

    @classmethod
    def get_user_config_dir(cls) -> Path:
        return get_settings().MOBILERUN_USER_CONFIG_DIR

    @classmethod
    def get_user_config_path(cls) -> Path:
        return cls.get_user_config_dir() / cls.CONFIG_FILE

    @classmethod
    @contextmanager
    def override(cls, config: T) -> Iterator[None]:
        """Temporarily return this model from load() for this class and context."""
        token = cls._override.set(config)
        try:
            yield
        finally:
            cls._override.reset(token)

    @classmethod
    def load(cls, config_path: str | None = None) -> T:
        """Resolve an explicit path, contextual override, opt-in env path, or user file."""
        if config_path:
            return cls._load_user_config(Path(config_path))

        if config_path is None and (config := cls._override.get()) is not None:
            return config

        if cls.USE_ENV_PATH:
            env_config = get_settings().MOBILERUN_CONFIG
            if env_config and env_config.strip():
                return cls._load_user_config(Path(env_config))

        user_config_path = cls.get_user_config_path()

        if user_config_path.exists():
            return cls._load_user_config(user_config_path)

        config = cls.config_type()
        if cls.WRITE_DEFAULTS:
            cls.save(config)
        return config

    @classmethod
    def _load_user_config(
        cls, user_config_path: Path, *, edit: Callable[[T], None] | None = None
    ) -> T:
        old_version = None
        try:
            user_config_path = user_config_path.resolve()
            original = user_config_path.read_bytes()
            user_dict = cls.parse_yaml(original.decode("utf-8"))
            if not isinstance(user_dict, dict):
                raise ValueError("config must be a mapping")
            if cls.CONFIG_VERSION is not None:
                old_version = user_dict.get("_version", 1)
                if type(old_version) is not int or old_version < 1:
                    raise ValueError("_version must be a positive integer")
                user_dict["_version"] = old_version
                user_dict = cls.migrate_dict(user_dict)
            config = cls.from_dict(user_dict)
        except (
            OSError,
            UnicodeError,
            yaml.YAMLError,
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            if cls._invalid_config_error is None:
                raise
            raise cls._invalid_config_error(
                f"Invalid config at {user_config_path}: {exc}"
            ) from exc

        # Do not publish a migration if the caller's edit fails.
        if edit is not None:
            edit(config)
        if old_version is not None and user_dict.get("_version", 0) > old_version:
            backup = user_config_path.with_name(
                f"{user_config_path.name}.v{old_version}.bak"
            )
            cls._save_backup(backup, original)
            if edit is None:
                cls._save_dict(user_dict, user_config_path)

        return config

    @staticmethod
    def _save_backup(backup: Path, original: bytes) -> None:
        # Publish a complete, private backup without replacing an older one.
        target = NamedTemporaryFile(dir=backup.parent, delete=False)
        try:
            with target:
                target.write(original)
                target.flush()
                os.fsync(target.fileno())
            try:
                os.link(target.name, backup)
            except FileExistsError:
                if not stat.S_ISREG(backup.lstat().st_mode):
                    raise FileExistsError(
                        f"Migration backup must be a regular file, not a link or special file: {backup}"
                    ) from None
                if backup.read_bytes() != original:
                    raise
        finally:
            Path(target.name).unlink(missing_ok=True)

    @classmethod
    def update(cls, edit: Callable[[T], None]) -> Path:
        """Read, edit in place, and save the user file under an interprocess lock."""
        path = cls.get_user_config_path()
        with cls._lock(path):
            if path.exists():
                config = cls._load_user_config(path, edit=edit)
            else:
                config = cls.config_type()
                edit(config)
            return cls._save_config(config, path)

    @classmethod
    def save(cls, config: T) -> Path:
        """Atomically replace the user file; prefer update for read-modify-write."""
        path = cls.get_user_config_path()
        with cls._lock(path):
            return cls._save_config(config, path)

    @staticmethod
    def _lock(path: Path) -> FileLock:
        path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(f"{path}.lock")

    @classmethod
    def _save_config(cls, config: T, path: Path) -> Path:
        config_dict = cls.to_dict(config)
        if cls.CONFIG_VERSION is not None:
            config_dict["_version"] = cls.CONFIG_VERSION
        return cls._save_dict(config_dict, path)

    @classmethod
    def _save_dict(cls, config_dict: dict[str, Any], path: Path) -> Path:
        """Save config dict to path."""
        destination = path.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        f = NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        )
        try:
            with f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(f.name, destination)
        finally:
            Path(f.name).unlink(missing_ok=True)
        return path


class ConfigLoader(ConfigFile[MobileConfig]):
    """The shared config.yaml file and its Mobilerun migrations."""

    APP_NAME = "droidrun"
    CONFIG_FILE = "config.yaml"
    config_type = MobileConfig
    WRITE_DEFAULTS = True
    USE_ENV_PATH = True
    CONFIG_VERSION = CURRENT_VERSION
    _invalid_config_error = OutdatedConfigError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MobileConfig:
        return cls.config_type.from_dict(data)

    @classmethod
    def to_dict(cls, config: MobileConfig) -> dict[str, Any]:
        return config.to_dict()

    @classmethod
    def parse_yaml(cls, text: str) -> object:
        # Unlike new config files, config.yaml rejects an empty document.
        return yaml.safe_load(text)

    @classmethod
    def migrate_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        return migrate(data, for_storage=True)

    @classmethod
    def save(cls, config: MobileConfig) -> Path:
        """Replace config.yaml without creating a lock file; use update to lock edits."""
        return cls._save_config(config, cls.get_user_config_path())
