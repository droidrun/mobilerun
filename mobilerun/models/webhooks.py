"""Webhook endpoint metadata safe to display in diagnostics."""

from dataclasses import dataclass
from typing import Literal

from pydantic import SecretStr


@dataclass(frozen=True)
class WebhookEndpoint:
    """A destination, not proof of ownership. A signing secret is returned only on creation."""

    id: str
    kind: Literal["http", "integration"]
    url: SecretStr | None
    state: Literal["ACTIVE", "DISABLED", "DELETED"]
    health: Literal["healthy", "failing", "blocked"]
    event_types: tuple[str, ...]
    secret: SecretStr | None = None
