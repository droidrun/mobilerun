"""Read and update the saved default without discarding other config keys."""

from pathlib import Path

import yaml

from mobilerun.config.loader import ConfigLoader
from mobilerun.config.settings import get_settings


def save_default_device(serial: str | None, *, config_path: str | None = None) -> None:
    ConfigLoader.load(config_path)
    if config_path:
        path = Path(config_path)
    else:
        configured_path = get_settings().MOBILERUN_CONFIG
        path = (
            Path(configured_path)
            if configured_path and Path(configured_path).exists()
            else ConfigLoader.get_user_config_path()
        )
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    device = data.setdefault("device", {})
    if serial is None:
        device.pop("serial", None)
    else:
        device["serial"] = serial
    ConfigLoader._save_dict(data, path)
