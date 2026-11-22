"""Rotating residential Mobilerun Connect proxy selection."""

from typing import Self

from mobilerun.config.settings import get_settings
from mobilerun.core.proxy.base import CloudProxy
from mobilerun.core.proxy.target import ProxyStatus, ProxyTarget


class CloudRotatingProxy(CloudProxy):
    def __init__(
        self, country: str, smart_ip: bool = True, *, name: str | None = None
    ) -> None:
        super().__init__(name=name)
        if not country:
            raise ValueError("A country is required for a rotating proxy")
        self.country = country
        self.smart_ip = smart_ip

    @classmethod
    def _from_settings(cls) -> Self:
        settings = get_settings()
        return cls(
            country=settings.MOBILERUN_CLOUD_PROXY_COUNTRY,
            smart_ip=settings.MOBILERUN_CLOUD_PROXY_SMART_IP,
        )

    def _parameters(self) -> dict[str, object]:
        return {
            "connect": {"country": self.country},
            "smart_ip": self.smart_ip,
        }

    async def _apply(self, target: ProxyTarget) -> ProxyStatus:
        return await target.connect_rotating(
            country=self.country, smart_ip=self.smart_ip
        )
