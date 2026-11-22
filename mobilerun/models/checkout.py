"""Public checkout details for a pending cloud resource."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Checkout:
    url: str
    expires_at: datetime | None
    resource: str
    resource_id: str
