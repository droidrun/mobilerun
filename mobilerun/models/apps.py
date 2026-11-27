"""Application value type shared by devices and clients."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class App:
    package: str
    version: str
    name: str
    id: str | None = None
    version_code: int | None = None
    path: Path | None = None
