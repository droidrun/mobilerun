"""Read-only remote observation; callers hold the shared registry lock."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from mobilerun.core.extensions.registry import Binding
from mobilerun.core.proxy.cloud_fixed import CloudFixedProxy
from mobilerun.core.proxy.existing import ExistingCloudProxy
from mobilerun.core.proxy.target import ProxyStatus

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device
    from mobilerun.core.proxy.base import CloudProxy

_READBACK_TIMEOUT = 10.0


async def _proxy_id(device: Device, status: ProxyStatus) -> str | None:
    reference = status.name or ""
    if reference.startswith("mrc:"):
        return reference[4:].strip() or None
    if not reference:
        return None
    return await device._proxy_target._resolve_reference(reference)


async def _adopt_remote_proxy(
    device: Device, *, name: str = "proxy", restored: CloudProxy | None = None
) -> None:
    deadline = asyncio.get_running_loop().time() + _READBACK_TIMEOUT
    async with asyncio.timeout_at(deadline):
        status = await device._proxy_target.status()
    registry = device._extension_state
    previous = tuple(
        binding
        for binding in registry.bindings.values()
        if binding.extension.exclusive == "proxy"
    )
    observed: CloudProxy | None = None
    if status.connected:
        try:
            async with asyncio.timeout_at(deadline):
                proxy_id = await _proxy_id(device, status)
        except Exception:
            # A failed identity read does not invalidate a successful status
            # read: the proxy is known active, only its identity is unknown.
            proxy_id = None
        if restored is not None:
            applied = restored._last_applied_status
            reference = (applied.name or "").removeprefix("mrc:") if applied else ""
            observed_reference = (status.name or "").removeprefix("mrc:")
            if not status.name or (
                reference and reference in (proxy_id, observed_reference)
            ):
                # The restore hook succeeded and read-back has no contradictory
                # identity. Retain the user's handle and all binding metadata.
                return
        observed = (
            CloudFixedProxy(proxy_id, name=name)
            if proxy_id is not None
            else ExistingCloudProxy(name=name)
        )
        observed._check_name(device, tuple(binding.extension for binding in previous))
        observed._action_entries()
    # Publish only after status/identity reads and validation succeeded. Never
    # invoke bind/unbind hooks: observation must not switch the remote proxy.
    for binding in previous:
        registry.bindings.pop(binding.extension.name)
        binding.extension._binding = None
    if observed is not None:
        binding = Binding(observed, device, False)
        registry.bindings[observed.name] = binding
        observed._binding = binding
