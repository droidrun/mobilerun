"""In-memory device for local and cloud proxy extension tests."""

from unittest.mock import AsyncMock

from mobilerun.core.proxy.target import ProxyTarget
from mobilerun.models.proxy import ProxyStatus
from tests.support.devices import MemoryDevice

__all__ = ["ProxyMemory"]


class FakeTarget(ProxyTarget):
    def __init__(self):
        self.current = ProxyStatus(False)
        self.connect = AsyncMock(side_effect=self._connect)
        self.disconnect_calls = 0

    async def _connect(self, name):
        self.current = ProxyStatus(True, name, "socks5")
        return self.current

    async def connect_fixed(self, *, id):
        return await self.connect(id)

    async def connect_rotating(self, *, country, smart_ip):
        return await self.connect(country)

    async def connect_socks5(self, host, port, user, password):
        return await self.connect(host)

    async def set_http_proxy(self, host, port):
        return await self.connect(f"{host}:{port}")

    async def status(self):
        return self.current

    async def disconnect(self):
        self.disconnect_calls += 1
        self.current = ProxyStatus(False)


class ProxyMemory(MemoryDevice):
    def __init__(self, cloud=False):
        super().__init__(
            cloud_id="cloud" if cloud else None,
            supported_operations=frozenset(
                {"proxy_switch"} if cloud else {"http_proxy"}
            ),
        )
        self.target = FakeTarget()

    @property
    def _proxy_target(self):
        return self.target
