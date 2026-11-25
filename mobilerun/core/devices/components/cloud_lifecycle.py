"""Cloud phone creation, resource ownership and lifecycle."""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType
from typing import Any, Literal, Self

from mobilerun.clients.cloud import devices
from mobilerun.clients.cloud.client import AsyncMobilerun
from mobilerun.config.settings import get_settings
from mobilerun.core.devices.base import Device
from mobilerun.core.phone_number.cloud import CloudPhoneNumber, _BorrowedNumberBinding
from mobilerun.core.proxy import CloudRotatingProxy
from mobilerun.models.billing import Billing, Plan

logger = logging.getLogger(__name__)


def _create_params(params: dict[str, Any]) -> dict[str, Any]:
    params = dict(params)
    if "country" in params:
        params["body_country"] = params.pop("country")
    proxy = params.pop("proxy", CloudRotatingProxy._from_settings())
    if proxy is None:
        raise ValueError("A proxy is required to create a cloud phone")
    params["proxy"] = proxy._parameters()
    return params


class _CloudLifecycle(Device):
    device_id: str | None
    _keep: bool
    _ready: bool
    _owned: bool
    _number_loaded: bool
    _client: AsyncMobilerun
    _owned_client: AsyncMobilerun | None
    _billing: Billing
    _plan: Plan | str | None
    _device_type: Literal["android_cloud_phone", "ios_phone", "device_slot"] | None
    _connect_lock: asyncio.Lock
    _create_options: dict[str, Any]
    _cloud_telephony: bool
    _cloud_proxy: bool | None

    @property
    def _cloud_client(self) -> AsyncMobilerun:
        return self._client

    def _number_cloud_client(self) -> AsyncMobilerun:
        return self._client

    def _require_device_id(self) -> str:
        if self.device_id is None:
            raise RuntimeError("Await CloudDevice(...) before using a new cloud phone")
        return self.device_id

    @property
    def _keep_policy(self) -> bool:
        return self._keep

    async def _ensure_connected(self) -> None:
        if not self._ready:
            raise RuntimeError(
                "Await CloudDevice(...) or use async with before using a new cloud phone"
            )

    async def _load_number(self) -> None:
        response = await devices.number(
            self._require_device_id(), billing=self._billing, client=self._client
        )
        binding = response.binding
        if binding:
            number = CloudPhoneNumber(
                binding.number_id, client=self._client, billing=self._billing
            )
            number._device_id = self.device_id
            number._phone_number = binding.phone_number
            number._binding_loaded = True
            async with self._extension_state.lock:
                number._check_name(self)
                number._action_entries()
                adopted = _BorrowedNumberBinding(number, self, False)
                self._extension_state.bindings[number.name] = adopted
                number._binding = adopted
        self._number_loaded = True

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._ready:
                info = await devices.capabilities(
                    self._require_device_id(),
                    billing=self._billing,
                    client=self._client,
                )
                self._cloud_telephony = bool(info.capabilities.telephony)
                self._cloud_proxy = info.capabilities.proxy
                if not self._number_loaded:
                    if self._cloud_telephony:
                        await self._load_number()
                    else:
                        self._number_loaded = True
                return
            if self.device_id is None:
                device_type = (
                    self._device_type
                    if self._device_type is not None
                    else get_settings().MOBILERUN_CLOUD_DEVICE_TYPE
                )
                if device_type not in (
                    "android_cloud_phone",
                    "ios_phone",
                    "device_slot",
                ):
                    raise ValueError(f"Invalid cloud device type: {device_type}")
                plan_options = {"plan": self._plan} if self._plan is not None else {}
                created = await devices.create(
                    client=self._client,
                    billing=self._billing,
                    device_type=device_type,
                    **plan_options,
                    **self._create_options,
                )
                self.device_id = created.id
                self._name = created.name
                self._owned = True
                logger.info("Created cloud device %s", self.device_id)
            try:
                await devices.wait_ready(
                    self.device_id, billing=self._billing, client=self._client
                )
                info = await devices.capabilities(
                    self.device_id, billing=self._billing, client=self._client
                )
                self._cloud_telephony = bool(info.capabilities.telephony)
                self._cloud_proxy = info.capabilities.proxy
                # Newly created phones cannot have a binding: creation has no number option.
                self._number_loaded = True
            except BaseException:
                if self._owned and not self._keep:
                    await devices.terminate(
                        self.device_id, billing=self._billing, client=self._client
                    )
                    self._owned = False
                    self.device_id = None
                raise
            self._ready = True

    async def _close_resources(self) -> None:
        pass

    async def close(self) -> None:
        async with self._closing_extensions():
            try:
                await self._close_resources()
            finally:
                try:
                    async with self._connect_lock:
                        if self._owned and not self._keep:
                            await devices.terminate(
                                self._require_device_id(),
                                billing=self._billing,
                                client=self._client,
                            )
                            self._owned = False
                finally:
                    if not self._owned or self._keep:
                        client = self._owned_client
                        self._owned_client = None
                        if client is not None:
                            await client.close()

    async def __aenter__(self) -> Self:
        return await self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def info(self) -> Any:
        await self._ensure_connected()
        info = await devices.retrieve(
            self._require_device_id(), billing=self._billing, client=self._client
        )
        self._name = info.name
        return info

    async def rename(self, name: str) -> Any:
        """Set the server display name and refresh the read-only name cache."""
        await self._ensure_connected()
        info = await devices.set_name(
            self._require_device_id(), name, billing=self._billing, client=self._client
        )
        self._name = info.name
        return info

    async def resume(self, **params: Any) -> Any:
        await self._ensure_connected()
        return await devices.resume(
            self._require_device_id(),
            billing=self._billing,
            client=self._client,
            **params,
        )

    async def stop(self, **params: Any) -> Any:
        await self._ensure_connected()
        return await devices.stop(
            self._require_device_id(),
            billing=self._billing,
            client=self._client,
            **params,
        )

    async def terminate(self, **params: Any) -> Any:
        await self._ensure_connected()
        await self._close_resources()
        result = await devices.terminate(
            self._require_device_id(),
            billing=self._billing,
            client=self._client,
            **params,
        )
        self._owned = False
        return result

    async def wait_ready(self, **params: Any) -> Any:
        await self._ensure_connected()
        return await devices.wait_ready(
            self._require_device_id(),
            billing=self._billing,
            client=self._client,
            **params,
        )

    async def reboot(self, **params: Any) -> Any:
        await self._ensure_connected()
        return await devices.reboot(
            self._require_device_id(),
            billing=self._billing,
            client=self._client,
            **params,
        )

    async def reset(self, **params: Any) -> Any:
        await self._ensure_connected()
        return await devices.reset(
            self._require_device_id(),
            billing=self._billing,
            client=self._client,
            **params,
        )
