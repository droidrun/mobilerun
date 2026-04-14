"""Migration v3: Add device.auto_setup field."""

from typing import Any, Dict

VERSION = 3


def migrate(config: Dict[str, Any]) -> Dict[str, Any]:
    """Add auto_setup to device config (defaults to True)."""
    device = config.setdefault("device", {})
    device.setdefault("auto_setup", True)
    return config
