"""Shared configuration; agent-owned sections are preserved as opaque data."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from os import PathLike
from typing import Any, Literal, Self

import yaml

from mobilerun.config.env import get_secrets, secret_value
from mobilerun.config.migrations import CURRENT_VERSION, migrate


@dataclass
class DeviceConfig:
    serial: str | None = None
    control_backend: str | None = None
    device_id: str = "auto"
    use_tcp: bool = False
    platform: str = "android"
    portal_mode: Literal["auto", "required", "disabled"] = "auto"
    auto_setup: bool = True
    auth_token: str | None = None

    def resolve_auth_token(self) -> str | None:
        return secret_value(get_secrets().MOBILERUN_DEVICE_TOKEN) or self.auth_token


@dataclass
class TelemetryConfig:
    enabled: bool = True


@dataclass
class TracingConfig:
    enabled: bool = False
    provider: str = "phoenix"
    langfuse_screenshots: bool = False
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = ""
    langfuse_user_id: str = "anonymous"
    langfuse_session_id: str = ""


@dataclass
class LoggingConfig:
    debug: bool = False
    level: Literal["quiet", "normal", "verbose", "debug"] = "normal"
    save_trajectory: str = "none"
    trajectory_path: str = "trajectories"
    rich_text: bool = False
    trajectory_gifs: bool = True


@dataclass
class MobileConfig:
    """Device settings plus lossless data for optional agent consumers."""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    agent: dict = field(default_factory=lambda: {"streaming": True})
    llm_profiles: dict = field(default_factory=dict)
    tools: dict = field(default_factory=dict)
    credentials: dict = field(default_factory=dict)
    external_agents: dict = field(default_factory=dict)
    mcp: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data = deepcopy(data)
        if "_version" in data and data["_version"] < CURRENT_VERSION:
            data = migrate(data)
        shared = {
            name: kind(**(data.get(name) or {}))
            for name, kind in (
                ("device", DeviceConfig),
                ("telemetry", TelemetryConfig),
                ("tracing", TracingConfig),
                ("logging", LoggingConfig),
            )
        }
        raw = {}
        for name in (
            "agent",
            "llm_profiles",
            "tools",
            "credentials",
            "external_agents",
            "mcp",
        ):
            value = data.get(name) or {}
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a mapping")
            raw[name] = value
        return cls(**shared, **raw)

    @classmethod
    def from_yaml(cls, path: str | PathLike[str]) -> Self:
        with open(path, encoding="utf-8") as stream:
            return cls.from_dict(yaml.safe_load(stream) or {})
