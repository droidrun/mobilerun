"""Validated host and port for device HTTP proxies."""

from __future__ import annotations

import asyncio
import ipaddress
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from mobilerun.core.extensions import Extension
from mobilerun.core.proxy.base import Proxy
from mobilerun.core.proxy.target import ProxyStatus, ProxyTarget
from mobilerun.errors import DeviceOperationNotSupported, ExtensionNotSupported

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device


class LocalProxy(Proxy):
    async def _reconcile_bind_failure(
        self, device: Device, failure: BaseException
    ) -> None:
        from mobilerun.core.extensions.registry import Binding

        try:
            async with asyncio.timeout(10):
                status = await device._proxy_target.status()
            observed = None
            if status.connected:
                host, _, port = (status.name or "").rpartition(":")
                observed = LocalProxy(host.strip("[]"), int(port), name=self.name)
            registry = device._extension_state
            previous = tuple(
                binding
                for binding in registry.bindings.values()
                if binding.extension.exclusive == "proxy"
            )
            for binding in previous:
                registry.bindings.pop(binding.extension.name)
                binding.extension._binding = None
            if observed is not None:
                binding = Binding(observed, device, False)
                registry.bindings[observed.name] = binding
                observed._binding = binding
        except (Exception, asyncio.CancelledError):
            self._mark_uncertain(device, failure)

    def __init__(self, host: str, port: int, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.host = host
        self.port = port
        try:
            ipaddress.ip_address(self.host)
        except ValueError:
            hostname = self.host.removesuffix(".")
            if (
                len(hostname) > 253
                or not hostname
                or not all(
                    re.fullmatch(
                        r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label
                    )
                    for label in hostname.split(".")
                )
            ):
                raise ValueError("Invalid proxy host") from None
        if not 1 <= self.port <= 65535:
            raise ValueError("Invalid proxy port")

    async def check_bind(self, device: Device) -> Sequence[Extension]:
        conflicts = await super().check_bind(device)
        if (
            device.cloud_id is not None
            or device.platform.lower() != "android"
            or "http_proxy" not in device.supported_operations
        ):
            raise ExtensionNotSupported(
                "LocalProxy requires local Android http_proxy support"
            )
        return conflicts

    def __str__(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{host}:{self.port}"

    def _parameters(self) -> dict[str, object]:
        raise DeviceOperationNotSupported(
            "HTTP proxies aren't supported on cloud devices; use CloudRotatingProxy, CloudFixedProxy or CloudSocks5Proxy"
        )

    async def _apply(self, target: ProxyTarget) -> ProxyStatus:
        return await target.set_http_proxy(self.host, self.port)
