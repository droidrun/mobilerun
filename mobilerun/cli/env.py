"""Inspect resolved CLI environment values and their sources."""

from __future__ import annotations

import os

import click

from mobilerun.cli import deviceauth
from mobilerun.cli.options import (
    DEFAULT_API_URL,
    DEFAULT_AUTH_URL,
    Command,
    config_option,
    emit,
    examples,
    json_option,
    verbose_option,
)
from mobilerun.cli.selection import CLOUD_CREDENTIAL_LABELS, resolve_cloud_credential
from mobilerun.config_manager.env_keys import API_KEY_ENV_VARS, load_env_key_sources

TRUTHY = {"1", "true", "yes", "on"}
BOOLEAN_VARIABLES = {"MOBILERUN_DEBUG", "MOBILERUN_STREAM_SCREENSHOTS", "MOBILERUN_OAUTH_MANUAL"}


def resolved_environment(config_path=None):
    from mobilerun.config_manager import ConfigLoader, MobileConfig
    from mobilerun.config_manager.config_manager import PORTAL_TOKEN_ENV_VARS
    from mobilerun.telemetry.tracker import is_telemetry_enabled

    path = ConfigLoader.resolve_path(config_path)
    exists = path.exists()
    config = ConfigLoader.load(config_path) if exists else MobileConfig()
    config_source = "config" if exists else "default"
    ctx = click.get_current_context(silent=True)
    path_source = (
        "flag" if config_path and (ctx is None or ctx.get_parameter_source("config_path")
                                   == click.core.ParameterSource.COMMANDLINE)
        else "env" if os.environ.get("MOBILERUN_CONFIG") else "default"
    )
    cloud_credential = resolve_cloud_credential()
    credentials = deviceauth.load_credentials() or {}
    values = {
        "MOBILERUN_TARGET": ("local", "default", False),
        "MOBILERUN_DEVICE": (config.device.serial, config_source if config.device.serial else "not set", False),
        "MOBILERUN_API_KEY": (
            cloud_credential.value,
            cloud_credential.source,
            True,
        ),
        "MOBILERUN_API_BASE_URL": (DEFAULT_API_URL, "default", False),
        "MOBILERUN_AUTH_URL": (
            credentials.get("auth_url") or DEFAULT_AUTH_URL,
            "credential file" if credentials.get("auth_url") else "default",
            False,
        ),
        "MOBILERUN_CONFIG": (str(path), path_source, False),
        "MOBILERUN_DEBUG": (config.logging.debug, config_source, False),
        "MOBILERUN_TELEMETRY_ENABLED": (config.telemetry.enabled, config_source, False),
        "MOBILERUN_STREAM_SCREENSHOTS": (False, "default", False),
        "MOBILERUN_OAUTH_MANUAL": (False, "default", False),
        "ANDROID_SERIAL": (None, "default", False),
        "PHOENIX_URL": ("http://0.0.0.0:6006", "default", False),
        "PHOENIX_PROJECT_NAME": ("", "default", False),
        "NO_COLOR": (None, "default", False),
    }
    for platform, name in PORTAL_TOKEN_ENV_VARS.items():
        value = config.device.resolve_auth_token(platform)
        source = "env" if os.environ.get(name) else config_source if value else "not set"
        values[name] = (value, source, True)
    sources = load_env_key_sources()
    for slot, name in API_KEY_ENV_VARS.items():
        source = sources[slot]
        values[name] = (source.saved, "credential file" if source.saved else "default", True)
    for name in ("OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        values[name] = (None, "default", True)
    not_from_env = {"MOBILERUN_API_KEY"}
    if path_source == "flag":
        not_from_env.add("MOBILERUN_CONFIG")
    for name in values.keys() - not_from_env:
        if os.environ.get(name):
            values[name] = (os.environ[name], "env", values[name][2])
    if ctx and any(
        parent.get_parameter_source("verbose") == click.core.ParameterSource.COMMANDLINE
        for parent in (ctx, ctx.find_root())
    ):
        values["MOBILERUN_DEBUG"] = (ctx.params.get("verbose"), "flag", False)
    values["MOBILERUN_TELEMETRY_ENABLED"] = (
        is_telemetry_enabled(config_enabled=config.telemetry.enabled),
        values["MOBILERUN_TELEMETRY_ENABLED"][1], False,
    )
    rows = []
    for name, (value, source, secret) in values.items():
        if name in BOOLEAN_VARIABLES and isinstance(value, str):
            value = value.lower() in TRUTHY
        if secret and not value and source is not None:
            source = "not set"
        if secret:
            value = "set" if value else "not set"
        rows.append({"variable": name, "value": value, "source": source})
    return rows


@click.command(cls=Command, epilog=examples('mobilerun env -j'))
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
            value = row["value"]
            if isinstance(value, bool):
                value = str(value).lower()
            elif value is None:
                value = ""
            label = (CLOUD_CREDENTIAL_LABELS[row["source"]]
                     if row["variable"] == "MOBILERUN_API_KEY" else row["source"])
            show_source = not (value == "not set" and label == "not set")
            source = f"  ({label})" if show_source else ""
            emit(f"{row['variable']}={value}{source}")
