"""Routine declarations and shared sandbox contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, ParamSpec, TypeVar

from mobilerun.config.settings import get_settings
from mobilerun.models.billing import Billing

from .errors import SandboxDeclarationError

P = ParamSpec("P")
R = TypeVar("R")


def routine(func: Callable[P, R]) -> Callable[P, R]:
    """Register a synchronous or asynchronous routine with upstream's runner.

    Use ``ctx.device()`` for the upstream synchronous device when needed.
    """
    from ._runtime import script

    return script(func)


JSON = dict[str, Any]


def validate_device(device: str | None) -> None:
    if device is not None and (not isinstance(device, str) or not device.strip()):
        raise SandboxDeclarationError("Device must be a nonempty string")


@dataclass(frozen=True)
class CloudDeclaration:
    name: str
    schedule: str | None = None
    device: str | None = None
    endpoint: bool | None = None

    @property
    def has_endpoint(self) -> bool:
        """Resolve the endpoint default without overriding another trigger."""
        return self.endpoint is True or (
            self.endpoint is None and self.schedule is None
        )

    def validate(self) -> None:
        """Reject invalid or competing declaration options."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise SandboxDeclarationError("Cloud name must be a nonempty string")
        if self.schedule is not None and (
            not isinstance(self.schedule, str) or len(self.schedule.split()) != 5
        ):
            raise SandboxDeclarationError("Schedule must have exactly five cron fields")
        validate_device(self.device)
        if self.endpoint is not None and not isinstance(self.endpoint, bool):
            raise SandboxDeclarationError("Endpoint must be a boolean or None")
        if self.schedule is not None and self.endpoint is True:
            raise SandboxDeclarationError("Choose either a schedule or an endpoint")


@dataclass(frozen=True)
class Artifact:
    declaration: CloudDeclaration
    source: str
    digest: str
    input_schema: JSON
    output_schema: JSON | None


@dataclass(frozen=True)
class DeploymentPreview:
    name: str
    device: str | None
    schedule: str | None
    billing: Billing


@dataclass(frozen=True)
class Endpoint:
    url: str
    trigger_id: str


@dataclass(frozen=True)
class DeploymentReceipt:
    name: str
    function_id: str
    version: int
    status: Literal["created", "pushed", "unchanged"]
    flow_id: str | None
    trigger_id: str | None
    action_ids: tuple[str, ...]
    digest: str
    endpoint: Endpoint | None
    warnings: tuple[str, ...]

    def to_dict(self) -> JSON:
        """Serialize the receipt's explicit fields."""
        return {
            "name": self.name,
            "function_id": self.function_id,
            "version": self.version,
            "status": self.status,
            "flow_id": self.flow_id,
            "trigger_id": self.trigger_id,
            "action_ids": list(self.action_ids),
            "digest": self.digest,
            "endpoint": (
                {"url": self.endpoint.url, "trigger_id": self.endpoint.trigger_id}
                if self.endpoint is not None
                else None
            ),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class OwnershipReceipt:
    origin: str
    account_id: str
    name: str
    function_id: str | None
    flow_id: str | None
    trigger_id: str | None
    action_ids: tuple[str, ...]
    digest: str | None
    stages: tuple[str, ...]

    def to_dict(self) -> JSON:
        """Serialize only the explicit ownership fields, never credentials."""
        return {
            "origin": self.origin,
            "account_id": self.account_id,
            "name": self.name,
            "function_id": self.function_id,
            "flow_id": self.flow_id,
            "trigger_id": self.trigger_id,
            "action_ids": list(self.action_ids),
            "digest": self.digest,
            "stages": list(self.stages),
        }


def resolve_billing(billing: Billing | str | None) -> Billing:
    """Resolve explicit billing before the shared environment default."""
    return Billing(billing) if billing is not None else get_settings().MOBILERUN_BILLING
