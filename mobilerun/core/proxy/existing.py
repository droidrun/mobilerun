"""Read-only representation of an active but unidentified cloud proxy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from mobilerun.core.proxy.base import CloudProxy
from mobilerun.core.proxy.target import ProxyStatus, ProxyTarget
from mobilerun.errors import ExtensionConflict
from mobilerun.models.context import ContextItem

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device


class ExistingCloudProxy(CloudProxy):
    """An observed existing proxy, not credentials or provisionable capacity."""

    async def on_bind(self, device: Device) -> None:
        from mobilerun.core.proxy.adoption import _proxy_id

        status = await device._proxy_target.status()
        if not status.connected or await _proxy_id(device, status) is not None:
            raise ExtensionConflict(
                "The unidentified existing proxy is no longer active"
            )

    async def _apply(self, target: ProxyTarget) -> ProxyStatus:
        raise ExtensionConflict(
            "An unidentified existing proxy cannot be provisioned or restored"
        )

    def _parameters(self) -> dict[str, object]:
        raise ExtensionConflict(
            "An unidentified existing proxy cannot be used for cloud creation"
        )

    async def context(self) -> Sequence[ContextItem]:
        return (
            ContextItem(
                self.name, "proxy", {"description": "existing proxy (unknown)"}
            ),
        )
