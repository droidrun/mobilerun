"""iOS devices, detecting the portal protocol once during connection."""

import asyncio

import httpx
from mobilerun_core_local.driver.ios import (
    IOSPortalDriver,
    IOSPortalHttpDriver,
    detect_ios_portal_kind,
)

from mobilerun.core.devices.base import Device
from mobilerun.core.devices.components.identifiers import _sanitize_identifier
from mobilerun.core.devices.components.raw import _BackendIO
from mobilerun.core.operator import Operator
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.options import (
    StateProviderFactory,
    StateProviderOptions,
    android_state_provider,
    ios_state_provider,
)
from mobilerun.errors import DeviceNotReady, DeviceOperationNotSupported


class _DetectionTransport(httpx.AsyncBaseTransport):
    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        pass  # Detection must not close the device-owned transport.


class IOSDevice(_BackendIO, Device):
    async def uninstall_app(self, package: str) -> str:
        raise DeviceOperationNotSupported("iOS does not support uninstall")

    def __init__(
        self,
        url: str,
        bundle_identifiers: list[str] | None = None,
        *,
        token: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        state_provider: StateProvider | None = None,
        state_provider_factory: StateProviderFactory | None = None,
        state_options: StateProviderOptions | None = None,
        operator: Operator | None = None,
    ) -> None:
        self._url = url
        self._token = token
        self._transport = transport
        self._bundle_identifiers = bundle_identifiers
        self._driver: IOSPortalDriver | IOSPortalHttpDriver | None = None
        self._portal_http = False
        self._connect_lock = asyncio.Lock()
        super().__init__(
            state_provider=state_provider,
            state_provider_factory=state_provider_factory,
            state_options=state_options,
            operator=operator,
        )

    @property
    def _io_driver(self) -> IOSPortalDriver | IOSPortalHttpDriver:
        if self._driver is None:
            raise DeviceNotReady("Connect IOSDevice or use async with before using it")
        return self._driver

    @property
    def platform(self) -> str:
        return "ios"

    @property
    def identifier(self) -> str | None:
        return _sanitize_identifier(self._url)

    @property
    def supported_operations(self) -> set[str]:
        return self._io_driver.supported

    @property
    def supported_buttons(self) -> set[str]:
        return self._io_driver.supported_buttons

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._driver is None:
                detection = (
                    _DetectionTransport(self._transport)
                    if self._transport is not None
                    else None
                )
                kind = await detect_ios_portal_kind(
                    self._url, token=self._token, transport=detection
                )
                self._portal_http = kind == "portal-http"
                if self._portal_http:
                    self._driver = IOSPortalHttpDriver(
                        self._url, self._token, transport=self._transport
                    )
                else:
                    self._driver = IOSPortalDriver(
                        self._url, self._bundle_identifiers, transport=self._transport
                    )
            await self._ensure_connected()

    def _create_state_provider(
        self,
        options: StateProviderOptions | None = None,
        *,
        device: Device | None = None,
    ) -> StateProvider:
        if self._driver is None:
            raise DeviceNotReady(
                "Connect IOSDevice before resolving its state provider"
            )
        factory = android_state_provider if self._portal_http else ios_state_provider
        return factory(device or self, options or StateProviderOptions())

    async def close(self) -> None:
        async with self._closing_extensions():
            if self._driver is not None and self._driver._client is not None:
                await self._driver._client.aclose()
                self._driver._client = None
