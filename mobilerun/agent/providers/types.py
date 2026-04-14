from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderVariantSpec:
    """Internal provider runtime variant for a user-facing provider family."""

    id: str
    runtime_provider_name: str
    auth_mode: str
    default_model: str | None
    models: tuple[str, ...]
    requires_api_key: bool = False
    requires_base_url: bool = False
    credential_path: str | None = None
    runtime_transport_provider_name: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class ProviderFamilySpec:
    """User-facing provider family shown during setup."""

    id: str
    display_name: str
    variants: tuple[ProviderVariantSpec, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)
