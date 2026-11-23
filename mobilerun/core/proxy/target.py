"""Typed proxy application boundary, independent of device implementations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mobilerun.clients.cloud.proxies import ProxyConnectParams

from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.models.proxy import ProxyStatus


class ProxyTarget:
    async def _resolve_reference(self, reference: str) -> str | None:
        return None

    async def status(self) -> ProxyStatus:
        raise DeviceOperationNotSupported("Proxy status is unsupported")

    async def disconnect(self) -> None:
        raise DeviceOperationNotSupported("Proxy disconnect is unsupported")

    async def connect_rotating(self, *, country: str, smart_ip: bool) -> ProxyStatus:
        raise DeviceOperationNotSupported(
            "CloudRotatingProxy isn't supported on this device"
        )

    async def connect_fixed(self, *, id: str) -> ProxyStatus:
        raise DeviceOperationNotSupported(
            "CloudFixedProxy isn't supported on this device"
        )

    async def connect_socks5(
        self,
        host: str,
        port: int,
        user: str | None,
        password: str | None,
    ) -> ProxyStatus:
        raise DeviceOperationNotSupported(
            "CloudSocks5Proxy isn't supported on this device"
        )

    async def set_http_proxy(self, host: str, port: int) -> ProxyStatus:
        raise DeviceOperationNotSupported("LocalProxy isn't supported on this device")


class CloudProxyTarget(ProxyTarget):
    def __init__(
        self,
        connect: Callable[[ProxyConnectParams], Awaitable[ProxyStatus]],
        status: Callable[[], Awaitable[ProxyStatus]] | None = None,
        disconnect: Callable[[], Awaitable[None]] | None = None,
        *,
        resolve_reference: Callable[[str], Awaitable[str | None]] | None = None,
    ) -> None:
        self._connect = connect
        self._status = status
        self._disconnect = disconnect
        self._resolve = resolve_reference

    async def _resolve_reference(self, reference: str) -> str | None:
        if self._resolve is None:
            return None
        return await self._resolve(reference)

    async def status(self) -> ProxyStatus:
        if self._status is None:
            return await super().status()
        return await self._status()

    async def disconnect(self) -> None:
        if self._disconnect is None:
            return await super().disconnect()
        await self._disconnect()

    async def connect_rotating(self, *, country: str, smart_ip: bool) -> ProxyStatus:
        return await self._connect(
            {"connect": {"country": country}, "smart_ip": smart_ip}
        )

    async def connect_fixed(self, *, id: str) -> ProxyStatus:
        return await self._connect({"connect": {"id": id}})

    async def set_http_proxy(self, host: str, port: int) -> ProxyStatus:
        raise DeviceOperationNotSupported(
            "HTTP proxies aren't supported on cloud devices; use CloudRotatingProxy, CloudFixedProxy or CloudSocks5Proxy"
        )

    async def connect_socks5(
        self,
        host: str,
        port: int,
        user: str | None,
        password: str | None,
    ) -> ProxyStatus:
        params: ProxyConnectParams = {"socks5": {"host": host, "port": port}}
        if user is not None:
            params["socks5"]["user"] = user
        if password is not None:
            params["socks5"]["password"] = password
        return await self._connect(params)


class AndroidProxyTarget(ProxyTarget):
    def __init__(
        self,
        set_http: Callable[[str, int], Awaitable[ProxyStatus]],
        status: Callable[[], Awaitable[ProxyStatus]] | None = None,
        disconnect: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._set_http = set_http
        self._status = status
        self._disconnect = disconnect

    async def status(self) -> ProxyStatus:
        if self._status is None:
            return await super().status()
        return await self._status()

    async def disconnect(self) -> None:
        if self._disconnect is None:
            return await super().disconnect()
        await self._disconnect()

    async def connect_rotating(self, *, country: str, smart_ip: bool) -> ProxyStatus:
        raise DeviceOperationNotSupported("Android local supports only LocalProxy")

    async def connect_fixed(self, *, id: str) -> ProxyStatus:
        raise DeviceOperationNotSupported("Android local supports only LocalProxy")

    async def connect_socks5(
        self, host: str, port: int, user: str | None, password: str | None
    ) -> ProxyStatus:
        raise DeviceOperationNotSupported("Android local supports only LocalProxy")

    async def set_http_proxy(self, host: str, port: int) -> ProxyStatus:
        return await self._set_http(host, port)
