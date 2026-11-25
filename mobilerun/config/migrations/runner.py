"""Config migration system."""

import importlib
import pkgutil
from pathlib import Path
from typing import Any, Dict, List

CURRENT_VERSION = 8


def get_migrations() -> List:
    """Discover and load all migration modules."""
    migrations = []
    migrations_dir = Path(__file__).parent

    for _, name, _ in pkgutil.iter_modules([str(migrations_dir)]):
        if name.startswith("v") and name[1:4].isdigit():
            module = importlib.import_module(f".{name}", package=__package__)
            # Discover versioned migration modules dynamically.
            if hasattr(module, "VERSION") and hasattr(module, "migrate"):
                migrations.append(module)

    return sorted(migrations, key=lambda m: m.VERSION)


def migrate(config: Dict[str, Any], *, for_storage: bool = False) -> Dict[str, Any]:
    """Upgrade config, omitting cwd-dependent path resolution for stored data."""
    version = config.get("_version", 0)

    if version >= CURRENT_VERSION:
        return config

    migrations = get_migrations()

    for migration in migrations:
        if migration.VERSION > version:
            if not (for_storage and getattr(migration, "READ_TIME_ONLY", False)):
                config = migration.migrate(config)
            config["_version"] = migration.VERSION

    return config
