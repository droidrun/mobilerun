"""Proxy value types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProxyStatus:
    connected: bool
    name: str | None = None
    protocol: str | None = None
