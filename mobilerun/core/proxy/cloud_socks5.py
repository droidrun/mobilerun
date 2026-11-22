"""SOCKS5 parameters; repr never exposes the password."""

from mobilerun.core.proxy.base import CloudProxy
from mobilerun.core.proxy.target import ProxyStatus, ProxyTarget


class CloudSocks5Proxy(CloudProxy):
    def __init__(
        self,
        host: str,
        port: int,
        user: str | None = None,
        password: str | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.host = host
        self.port = port
        self.user = user
        self.password = password

    def _parameters(self) -> dict[str, object]:
        result: dict[str, object] = {
            "socks5": {
                key: value
                for key, value in {
                    "host": self.host,
                    "port": self.port,
                    "user": self.user,
                    "password": self.password,
                }.items()
                if value is not None
            }
        }
        return result

    async def _apply(self, target: ProxyTarget) -> ProxyStatus:
        return await target.connect_socks5(
            self.host, self.port, self.user, self.password
        )
