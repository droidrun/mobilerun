"""Routine declarations and shared sandbox contracts."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, ParamSpec, TypeVar
from uuid import UUID

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
WEBHOOK_REGISTER_HINT = "mobilerun webhook register <url>"
FILES_RETURN_ERROR = "files require a non-optional Pydantic BaseModel return annotation"
FILE_UPLOAD_USES = "files.upload"
FILE_MIME_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
_FILE_FIELD_PART = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|0|[1-9][0-9]*")
_UNSAFE_FIELD_PARTS = {"__proto__", "prototype", "constructor"}


def is_result_path_part(part: str) -> bool:
    """Check a literal key in the worker's dot-separated template syntax."""
    return (
        bool(part)
        and part == part.strip().strip("\ufeff").strip()
        and not any(char in part for char in ".{}")
        and part.lower() not in _UNSAFE_FIELD_PARTS
    )


def webhook_endpoint_id(value: str) -> str:
    """Return a canonical endpoint ID without exposing invalid input."""
    if isinstance(value, str):
        try:
            return str(UUID(value))
        except ValueError:
            pass
    raise SandboxDeclarationError(
        f"webhook= takes a webhook endpoint ID. Register one first: {WEBHOOK_REGISTER_HINT}"
    )


def validate_device(device: str | None) -> None:
    if device is not None and (not isinstance(device, str) or not device.strip()):
        raise SandboxDeclarationError("Device must be a nonempty string")


def _utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def _has_control_chars(value: str) -> bool:
    return any(unicodedata.category(char) == "Cc" for char in value)


@dataclass(frozen=True)
class CloudDeclaration:
    name: str
    schedule: str | None = None
    device: str | None = None
    endpoint: bool | None = None
    webhook: str | None = field(default=None, repr=False)
    files: dict[str, str] | None = None
    files_folder: str | None = None

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
        if self.webhook is not None:
            webhook_endpoint_id(self.webhook)
            if not self.has_endpoint and self.schedule is None:
                raise SandboxDeclarationError("A webhook requires a flow trigger")
        if self.endpoint is not None and not isinstance(self.endpoint, bool):
            raise SandboxDeclarationError("Endpoint must be a boolean or None")
        if self.schedule is not None and self.endpoint is True:
            raise SandboxDeclarationError("Choose either a schedule or an endpoint")
        self._validate_files()

    def _validate_files(self) -> None:
        if self.files is not None and not isinstance(self.files, dict):
            raise SandboxDeclarationError("files must be a dict of filenames to fields")
        if not self.files:
            return
        if self.schedule is None and not self.has_endpoint:
            raise SandboxDeclarationError("files require a flow trigger")
        folder = self.files_folder
        if (
            not isinstance(folder, str)
            or not folder.removeprefix("/").strip()
            or ".." in folder
            or _has_control_chars(folder)
            or any(part in {"", "."} for part in folder.removeprefix("/").split("/"))
            # JavaScript's schema counts the leading slash and UTF-16 code units.
            or _utf16_length("/" + folder.removeprefix("/")) > 1024
        ):
            raise SandboxDeclarationError(
                "files_folder must be a safe folder path of at most 1024 characters including the leading slash"
            )
        seen = set()
        for filename, path in self.files.items():
            if (
                not isinstance(filename, str)
                or not 1 <= _utf16_length(filename) <= 255
                or filename != filename.strip().strip("\ufeff")
                or "/" in filename
                or "\\" in filename
                or _has_control_chars(filename)
            ):
                raise SandboxDeclarationError(
                    "file filename must be a safe single path segment"
                )
            if Path(filename).suffix.lower() not in FILE_MIME_TYPES:
                raise SandboxDeclarationError(
                    "unsupported file extension; use .jpg, .jpeg, or .png"
                )
            normalized = filename.casefold()
            if normalized in seen:
                raise SandboxDeclarationError(
                    "file filename collision (case-insensitive)"
                )
            seen.add(normalized)
            if not isinstance(path, str) or not all(
                _FILE_FIELD_PART.fullmatch(part) and is_result_path_part(part)
                for part in path.split(".")
            ):
                raise SandboxDeclarationError("invalid output field in files")


@dataclass(frozen=True)
class Artifact:
    declaration: CloudDeclaration
    source: str
    digest: str
    input_schema: JSON
    output_schema: JSON | None


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
