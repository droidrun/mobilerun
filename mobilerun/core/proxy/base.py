"""Standalone proxy extensions; compatibility belongs to the extension."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

from mobilerun.core.extensions import Extension, action, device_action
from mobilerun.core.proxy.target import ProxyTarget
from mobilerun.errors import (
    ExtensionConflict,
    ExtensionNotSupported,
    UnmanagedProxyConflict,
)
from mobilerun.models.context import ContextItem
from mobilerun.models.proxy import ProxyStatus

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device


class Proxy(Extension):
    default_name = "proxy"
    exclusive = "proxy"
    _last_applied_status: ProxyStatus | None = None

    async def on_bind(self, device: Device) -> None:
        self._last_applied_status = await self._apply(device._proxy_target)

    async def on_unbind(self, device: Device) -> None:
        await device._proxy_target.disconnect()

    def _mark_uncertain(self, device: Device, failure: BaseException) -> None:
        from mobilerun.core.extensions.registry import Binding

        registry = device._extension_state
        previous = [
            binding
            for binding in registry.bindings.values()
            if binding.extension.exclusive == "proxy"
        ]
        if not previous:
            binding = Binding(self, device, False, uncertain=True)
            registry.bindings[self.name] = binding
            self._binding = binding
        else:
            for binding in previous:
                binding.uncertain = True
        failure.add_note(
            "Remote proxy status could not be reconciled; binding is uncertain. Reconnect or bind another proxy with force=True to recover."
        )

    @device_action
    async def status(self, device: Device) -> ProxyStatus:
        """Return current device proxy connection state without credentials."""
        return await device._proxy_target.status()

    async def context(self) -> Sequence[ContextItem]:
        return (ContextItem(self.name, "proxy", {"kind": self.exclusive}),)

    async def _apply(self, target: ProxyTarget) -> ProxyStatus:
        raise NotImplementedError

    def _parameters(self) -> dict[str, object]:
        raise NotImplementedError


class CloudProxy(Proxy):
    """Shared type for cloud-device creation and cloud-only proxy checks."""

    cloud_native = True

    @action
    async def status(self) -> ProxyStatus:
        """Return remote status when attached, or disconnected when unbound."""
        device = self.device
        return (
            ProxyStatus(False)
            if device is None
            else await device._proxy_target.status()
        )

    def _can_switch(self, device: Device) -> bool:
        return "proxy_switch" in device.supported_operations

    async def _reconcile_bind_failure(
        self, device: Device, failure: BaseException
    ) -> None:
        if isinstance(failure, UnmanagedProxyConflict):
            # on_bind refused before applying anything; there is no remote
            # mutation to reconcile and the incoming extension remains unbound.
            return
        from mobilerun.core.proxy.adoption import _adopt_remote_proxy

        try:
            restored = (
                self
                if self._binding is not None and not self._binding.uncertain
                else None
            )
            await _adopt_remote_proxy(device, name=self.name, restored=restored)
        except (Exception, asyncio.CancelledError):
            self._mark_uncertain(device, failure)

    async def check_bind(self, device: Device) -> Sequence[Extension]:
        conflicts = await super().check_bind(device)
        if device.cloud_id is None:
            raise ExtensionNotSupported("Cloud proxies require a cloud device")
        return conflicts

    async def on_bind(self, device: Device) -> None:
        state = device._extension_state
        status = await device._proxy_target.status()
        if status.connected and not state.force:
            raise UnmanagedProxyConflict(
                "Device has an unmanaged active proxy; use force=True to take over"
            )
        await super().on_bind(device)

    async def _check_unbind(self, device: Device) -> None:
        state = device._extension_state
        if state.switching:
            return  # The incoming cloud proxy replaces upstream in one SDK call.
        if not state.closing:
            raise ExtensionConflict(
                "cloud phones need a proxy; bind another one with force=True to switch"
            )

    async def on_unbind(self, device: Device) -> None:
        await self._check_unbind(device)
        # Closing a borrowed cloud phone must not leave it proxy-less. Closing
        # the relationship leaves its server configuration intact.
