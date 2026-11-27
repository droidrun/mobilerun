"""File value types shared by workspace and device operations."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class WorkspaceFile:
    id: str
    name: str
    mime_type: str
    size_bytes: int
    created_at: datetime


@dataclass(frozen=True)
class DeviceFile:
    path: str
    name: str
    size_bytes: int | None
    modified_at: datetime | None
