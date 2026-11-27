"""Visual remote device interface."""

from typing import Any

from mobilerun_core_local.driver.visual_remote import VisualRemoteDriver

from mobilerun.core.devices.base import Device
from mobilerun.core.devices.components.identifiers import _sanitize_identifier
from mobilerun.core.devices.components.raw import _BackendIO
from mobilerun.core.operator import Operator
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.options import (
    StateProviderFactory,
    StateProviderOptions,
    screenshot_state_provider,
)
from mobilerun.errors import DeviceOperationNotSupported


class RemoteDevice(_BackendIO, Device):
    """Device backed by a visual remote driver."""

    _backend = VisualRemoteDriver

    async def list_apps(self, include_system: bool = False) -> list[dict[str, str]]:
        raise DeviceOperationNotSupported(
            "Visual remote does not expose an app inventory"
        )

    async def _get_date(self) -> str:
        raise DeviceOperationNotSupported("Visual remote does not expose device time")

    async def _get_ui_tree(self) -> dict[str, Any]:
        raise DeviceOperationNotSupported("Visual remote does not expose a UI tree")

    async def start_app(self, package: str, activity: str | None = None) -> str:
        await self._ensure_connected()
        if "start_app" not in self.supported_operations:
            raise DeviceOperationNotSupported(
                "This visual remote server does not support app launch"
            )
        return await super().start_app(package, activity)

    @property
    def platform(self) -> str:
        return self._driver.platform

    def __init__(
        self,
        *args: Any,
        state_provider: StateProvider | None = None,
        state_options: StateProviderOptions | None = None,
        state_provider_factory: StateProviderFactory | None = None,
        operator: Operator | None = None,
        **kwargs: Any,
    ) -> None:
        self._driver = VisualRemoteDriver(*args, **kwargs)
        super().__init__(
            state_provider=state_provider,
            operator=operator,
            state_options=state_options,
            state_provider_factory=state_provider_factory,
        )

    @property
    def _io_driver(self) -> VisualRemoteDriver:
        return self._driver

    @property
    def supported_operations(self) -> set[str]:
        return self._driver.supported

    @property
    def supported_buttons(self) -> set[str]:
        return self._driver.supported_buttons

    @property
    def _remote_identifier(self) -> str | None:
        return _sanitize_identifier(self._driver.device_id or self._driver.url)

    @property
    def _url(self) -> str:
        return self._driver.url

    @property
    def identifier(self) -> str | None:
        return self._remote_identifier

    def _create_state_provider(
        self,
        options: StateProviderOptions | None = None,
        *,
        device: Device | None = None,
    ) -> StateProvider:
        return screenshot_state_provider(
            device or self, options or StateProviderOptions()
        )

    async def close(self) -> None:
        async with self._closing_extensions():
            if self._driver._client is not None:
                await self._driver._client.aclose()
                self._driver._client = None
