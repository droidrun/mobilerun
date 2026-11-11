"""Inspect resolved CLI environment values and their sources."""

import os
from pathlib import Path

import click

from mobilerun.cli import deviceauth
from mobilerun.cli.auth import DEFAULT_AUTH_URL
from mobilerun.cli.options import (
    CONTEXT_SETTINGS,
    DEFAULT_API_URL,
    Command,
    config_option,
    emit,
    json_option,
    verbose_option,
)
from mobilerun.cli.selection import resolve_cloud_api_key
from mobilerun.config_manager import ConfigLoader, MobileConfig
from mobilerun.config_manager.env_keys import API_KEY_ENV_VARS, load_env_key_sources


def resolved_environment(config_path=None):
    path = Path(
        config_path or os.environ.get("MOBILERUN_CONFIG") or ConfigLoader.get_user_config_path()
    )
    exists = path.exists()
    config = ConfigLoader.load(str(path)) if exists or config_path or os.environ.get("MOBILERUN_CONFIG") else MobileConfig()
    config_source = "config" if exists else "default"
    credentials = deviceauth.load_credentials() or {}
    values = {
        "MOBILERUN_TARGET": ("local", "default", False),
        "MOBILERUN_DEVICE": (config.device.serial, config_source, False),
        "MOBILERUN_CLOUD_API_KEY": (
            resolve_cloud_api_key(),
            "credential file" if resolve_cloud_api_key() else "default",
            True,
        ),
        "MOBILERUN_API_BASE_URL": (DEFAULT_API_URL, "default", False),
        "MOBILERUN_AUTH_URL": (
            credentials.get("auth_url") or DEFAULT_AUTH_URL,
            "credential file" if credentials.get("auth_url") else "default",
            False,
        ),
        "MOBILERUN_CONFIG": (str(path), "default", False),
        "MOBILERUN_DEVICE_TOKEN": (config.device.auth_token, config_source, True),
        "MOBILERUN_DEBUG": (config.logging.debug, config_source, False),
        "MOBILERUN_TELEMETRY_ENABLED": (config.telemetry.enabled, config_source, False),
        "MOBILERUN_STREAM_SCREENSHOTS": (False, "default", False),
        "MOBILERUN_OAUTH_MANUAL": (False, "default", False),
        "ANDROID_SERIAL": (None, "default", False),
        "PHOENIX_URL": ("http://0.0.0.0:6006", "default", False),
        "PHOENIX_PROJECT_NAME": ("", "default", False),
        "NO_COLOR": (None, "default", False),
    }
    sources = load_env_key_sources()
    for slot, name in API_KEY_ENV_VARS.items():
        source = sources[slot]
        values[name] = (source.saved, "credential file" if source.saved else "default", True)
    for name in ("OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        values[name] = (None, "default", True)
    rows = []
    for name, (value, source, secret) in values.items():
        if os.environ.get(name):
            value, source = os.environ[name], "env"
        if secret:
            value = "set" if value else "not set"
        rows.append({"variable": name, "value": value, "source": source})
    return rows


@click.command(cls=Command, context_settings=CONTEXT_SETTINGS)
@config_option
@verbose_option
@json_option
def env(config_path, verbose, json_output):
    """Show resolved environment values (secrets are set/not set)."""
    rows = resolved_environment(config_path)
    if json_output:
        emit(rows, json_output=True)
    else:
        for row in rows:
            emit(f"{row['variable']}={row['value']}  ({row['source']})")
