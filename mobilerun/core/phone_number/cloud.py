"""Owned cloud numbers, with resource-side binding only."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Self, cast

from mobilerun.clients.cloud import devices, numbers
from mobilerun.clients.cloud.client import AsyncMobilerun
from mobilerun.clients.local.codes import wait_for_code
from mobilerun.config.settings import get_settings
from mobilerun.core.devices.base import Device
from mobilerun.core.extensions import Extension, action
from mobilerun.core.phone_number.base import PhoneNumber
from mobilerun.errors import (
    CloudError,
    ExtensionNotSupported,
    PaymentRequired,
    PurchaseInProgress,
)
from mobilerun.models.billing import Billing, ResourcePlan
from mobilerun.models.context import ContextItem
from mobilerun.clients.cloud.checkout import CheckoutMode, checkout_callback
from mobilerun.clients.cloud.payments import (
    CheckoutCallback,
    PaymentAdapter,
    PaymentSnapshot,
    wait_for_payment,
    wait_until_active,
)
from mobilerun.utils.codes import extract_code
from mobilerun.utils.timestamps import to_api_timestamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ObservedNumberBinding:
    number_id: str
    phone_number: str | None


class CloudPhoneNumber(PhoneNumber):
    cloud_native = True

    def __init__(
        self,
        number_id: str | None = None,
        *,
        country: str | None = None,
        billing: Billing | str | None = None,
        plan: ResourcePlan | str | None = None,
        label: str | None = None,
        keep: bool = False,
        on_checkout: CheckoutCallback | None = None,
        checkout: CheckoutMode = "error",
        payment_timeout: float = 600,
        name: str | None = None,
        **connection: Any,
    ) -> None:
        super().__init__(name=name)
        unknown = connection.keys() - {"client", "api_key", "base_url"}
        if unknown:
            raise TypeError(
                f"Unexpected connection options: {', '.join(sorted(unknown))}"
            )
        if number_id == "":
            raise ValueError("Resource id must not be empty")
        self._id = number_id
        self._billing = Billing(
            get_settings().MOBILERUN_BILLING if billing is None else billing
        )
        if plan is not None:
            ResourcePlan(plan)
        self._plan = plan
        self._connection = connection
        self._params: dict[str, Any] = {
            key: value
            for key, value in {"country": country, "label": label}.items()
            if value is not None
        }
        self._keep = keep
        self._on_checkout = checkout_callback(checkout, on_checkout)
        self._payment_timeout = payment_timeout
        self._owned = False
        self.resumed_existing = False
        self._phone_number = None
        self._device_id = None
        self._binding_loaded = False
        self._lock = asyncio.Lock()

    @property
    def id(self) -> str:
        if self._id is None:
            raise RuntimeError("Await CloudPhoneNumber(...) before using it")
        return self._id

    @property
    def phone_number(self) -> str | None:
        return self._phone_number

    @property
    def device_id(self) -> str | None:
        return self._device_id

    def __await__(self) -> Generator[Any, None, Self]:
        return self._ready().__await__()

    async def _ready(self) -> Self:
        async with self._lock:
            if self._id is None:
                try:
                    response = await numbers.create(
                        billing=self._billing,
                        **({"plan": self._plan} if self._plan is not None else {}),
                        **(
                            {
                                "on_checkout": self._on_checkout,
                                "payment_timeout": self._payment_timeout,
                            }
                            if self._on_checkout is not None
                            else {}
                        ),
                        **({"keep": True} if self._keep else {}),
                        **self._connection,
                        **self._params,
                    )
                except PurchaseInProgress as exc:
                    if self._on_checkout is None or exc.checkout is None:
                        raise
                    self._id = exc.resource_id
                    self.resumed_existing = True
                    async with asyncio.timeout(self._payment_timeout):
                        result = self._on_checkout(exc.checkout)
                        if inspect.isawaitable(result):
                            await result
                        await self.wait_until_paid(timeout=self._payment_timeout)
                    return self
                except PaymentRequired as exc:
                    self._id = exc.checkout.resource_id
                    self._owned = True
                    raise PaymentRequired(exc.checkout, self.wait_until_paid) from exc
                data = response.data
                self._id = data.number_id
                self._owned = True
                self._phone_number = None
                if self._on_checkout is not None and data.state == "awaiting_payment":
                    ready = (
                        await numbers.retrieve(
                            self.id, billing=self._billing, **self._connection
                        )
                    ).data
                    self._phone_number = ready.phone_number
                    data = ready
                self._device_id = None
            else:
                response = await numbers.retrieve(
                    self.id, billing=self._billing, **self._connection
                )
                data = response.data
                self._phone_number = data.phone_number
                binding = data.device_binding
                self._device_id = binding.device_id if binding else None
            self._binding_loaded = True
        if data.state == "awaiting_payment":
            await wait_for_payment(
                PaymentAdapter(
                    "phone number", self.id, self._payment_state, self.release
                ),
                PaymentSnapshot(data.state, data.checkout_url),
                resume=self.wait_until_paid,
                on_checkout=self._on_checkout,
                timeout=self._payment_timeout,
            )
        return self

    async def _payment_state(self) -> PaymentSnapshot:
        data = (
            await numbers.retrieve(self.id, billing=self._billing, **self._connection)
        ).data
        self._phone_number = data.phone_number
        return PaymentSnapshot(data.state, data.checkout_url)

    async def wait_until_paid(self, timeout: float = 600) -> Self:
        await wait_until_active(
            PaymentAdapter("phone number", self.id, self._payment_state, self.release),
            timeout=timeout,
        )
        return self

    async def __aenter__(self) -> Self:
        return await self

    async def wait_ready(self, *, timeout: float | None = None) -> Self:
        """Wait for GET number data.state=active; payment is never automated."""
        timeout = (
            get_settings().MOBILERUN_CLOUD_NUMBER_READY_TIMEOUT_SECONDS
            if timeout is None
            else timeout
        )
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        loop = asyncio.get_running_loop()
        started = loop.time()
        state = "unknown"
        try:
            async with asyncio.timeout(timeout):
                while True:
                    data = (
                        await numbers.retrieve(
                            self.id, billing=self._billing, **self._connection
                        )
                    ).data
                    state = data.state
                    await wait_for_payment(
                        PaymentAdapter(
                            "phone number", self.id, self._payment_state, self.release
                        ),
                        PaymentSnapshot(data.state, data.checkout_url),
                        resume=self.wait_until_paid,
                    )
                    logger.debug(
                        "Number %s readiness state=%s elapsed=%.1fs",
                        self.id,
                        state,
                        loop.time() - started,
                    )
                    if state == "active":
                        self._phone_number = data.phone_number
                        return self
                    if state != "provisioning":
                        raise CloudError(code="number_unready", status=state)
                    await asyncio.sleep(2)
        except TimeoutError as exc:
            raise TimeoutError(
                f"Number {self.id} did not become active within {timeout:g}s; last state={state}"
            ) from exc

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    @classmethod
    async def included_available(
        cls, country: str, *, billing: Billing | str | None = None, **connection: Any
    ) -> int:
        """Return a capacity snapshot, not a reservation."""
        billing = Billing(
            get_settings().MOBILERUN_BILLING if billing is None else billing
        )
        data = (await numbers.capacity(country, billing=billing, **connection)).data
        return 0 if data.status == "not_included" else data.remaining

    @classmethod
    async def list(
        cls, *, billing: Billing | str | None = None, **connection: Any
    ) -> list[Self]:
        billing = Billing(
            get_settings().MOBILERUN_BILLING if billing is None else billing
        )
        result = []
        for item in await numbers.list_numbers(billing=billing, **connection):
            number = cls(item.id, billing=billing, **connection)
            number._phone_number = item.phone_number
            binding = item.device_binding
            number._device_id = binding.device_id if binding else None
            number._binding_loaded = True
            result.append(number)
        return result

    @action()
    async def messages(
        self, *, after: datetime | None = None
    ) -> list[numbers.MessageItem]:
        """List inbound messages received by this number."""
        return await numbers.messages(
            self.id,
            billing=self._billing,
            after=to_api_timestamp(after) if after is not None else None,
            direction="inbound",
            **self._connection,
        )

    @action()
    async def send(self, to: str, text: str) -> numbers.MessageSendResponse:
        """Send an SMS message to a phone number."""
        return await numbers.send(
            self.id, to, text, billing=self._billing, **self._connection
        )

    @action()
    async def wait_for_code(
        self,
        *,
        sender: str | None = None,
        timeout: float = 60,
        after: datetime | None = None,
    ) -> str:
        """Wait for a verification code sent to this number."""
        after = after if after is not None else datetime.now(timezone.utc)

        async def fetch(sender: str | None) -> str | None:
            for message in await self.messages(after=after):
                if sender and not any(
                    sender.casefold() in (value or "").casefold()
                    for value in (message.detected_sender, message.peer_number)
                ):
                    continue
                code = extract_code(message.body)
                if code:
                    return code
            return None

        return await wait_for_code(fetch, timeout=timeout, sender=sender)

    async def check_bind(self, device: Device) -> tuple[Extension, ...]:
        if (
            device.cloud_id is None
            or "cloud_telephony" not in device.supported_operations
        ):
            raise ExtensionNotSupported("This device does not support cloud telephony")
        return tuple(await super().check_bind(device))

    async def on_bind(self, device: Device) -> None:
        if device.cloud_id is None:
            raise ExtensionNotSupported("This device does not support cloud telephony")
        client = cast(
            AsyncMobilerun,
            device._unwrap()._number_cloud_client() or self._connection.get("client"),
        )
        await devices.bind_number(
            device.cloud_id, self.id, billing=self._billing, client=client
        )
        self._device_id = device.cloud_id
        self._binding_loaded = True

    async def on_unbind(self, device: Device) -> None:
        if device.cloud_id is None:
            raise ExtensionNotSupported("This device does not support cloud telephony")
        client = cast(
            AsyncMobilerun,
            device._unwrap()._number_cloud_client() or self._connection.get("client"),
        )
        await devices.unbind_number(
            device.cloud_id, billing=self._billing, client=client
        )
        self._device_id = None

    def _matches_remote(self, observed: object) -> bool:
        return (
            isinstance(observed, _ObservedNumberBinding)
            and self.id == observed.number_id
        )

    def _observe_remote_binding(self, device: Device, observed: object | None) -> None:
        self._device_id = device.cloud_id if observed is not None else None
        if observed is not None:
            if not isinstance(observed, _ObservedNumberBinding):
                raise TypeError("Unexpected phone number binding observation")
            self._phone_number = observed.phone_number
            self._binding_loaded = True

    async def _reconcile_bind_failure(
        self, device: Device, failure: BaseException
    ) -> None:
        from mobilerun.core.extensions.registry import Binding

        registry = device._extension_state
        previous = [
            binding
            for binding in registry.bindings.values()
            if binding.extension.exclusive == self.exclusive
        ]
        try:
            if device.cloud_id is None:
                raise RuntimeError("Missing cloud device ID")
            client = cast(
                AsyncMobilerun,
                device._unwrap()._number_cloud_client()
                or self._connection.get("client"),
            )
            async with asyncio.timeout(10):
                response = await devices.number(
                    device.cloud_id, billing=self._billing, client=client
                )
            binding_data = response.binding
            observed = (
                _ObservedNumberBinding(
                    binding_data.number_id, binding_data.phone_number
                )
                if binding_data is not None
                else None
            )
            selected = None
            if observed:
                selected = next(
                    (b for b in previous if b.extension._matches_remote(observed)),
                    None,
                )
                if selected is None:
                    number = (
                        self
                        if self.id == observed.number_id
                        else CloudPhoneNumber(
                            observed.number_id,
                            billing=self._billing,
                            name=self.name,
                            **self._connection,
                        )
                    )
                    number._check_name(device, tuple(b.extension for b in previous))
                    selected = Binding(number, device, False)
                selected.extension._observe_remote_binding(device, observed)
                selected.uncertain = False
            for binding in previous:
                registry.bindings.pop(binding.extension.name, None)
                binding.extension._binding = None
                if binding is not selected:
                    binding.extension._observe_remote_binding(device, None)
            if selected is not None:
                registry.bindings[selected.extension.name] = selected
                selected.extension._binding = selected
        except (Exception, asyncio.CancelledError):
            if not previous:
                binding = Binding(self, device, False, uncertain=True)
                registry.bindings[self.name] = binding
                self._binding = binding
            else:
                for binding in previous:
                    binding.uncertain = True
            failure.add_note(
                "Remote phone number binding could not be reconciled; binding is uncertain."
            )

    async def context(self) -> tuple[ContextItem, ...]:
        return (ContextItem(self.name, "phone_number", {"number": self.phone_number}),)

    async def release(self) -> Any:
        await self.unbind()
        result = await numbers.delete(
            self.id, billing=self._billing, **self._connection
        )
        self._owned = False
        return result

    async def on_close(self) -> None:
        if self._owned and not self._keep:
            await numbers.delete(self.id, billing=self._billing, **self._connection)
            self._owned = False
