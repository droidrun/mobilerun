"""Select an existing purchased Mobilerun Connect proxy."""

from mobilerun.core.proxy.base import CloudProxy
from mobilerun.core.proxy.target import ProxyTarget
from mobilerun.models.proxy import ProxyStatus


class CloudFixedProxy(CloudProxy):
    def __init__(self, id: str, *, name: str | None = None) -> None:
        super().__init__(name=name)
        if not id:
            raise ValueError("An existing proxy id is required")
        self.id = id

    def _parameters(self) -> dict[str, object]:
        return {"connect": {"id": self.id}}

    async def _apply(self, target: ProxyTarget) -> ProxyStatus:
        return await target.connect_fixed(id=self.id)
