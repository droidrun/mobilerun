"""Cloud mailbox lifecycle and local-only device attachment."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Self

from mobilerun.clients.cloud import mailboxes
from mobilerun.clients.cloud.checkout import CheckoutMode, checkout_callback
from mobilerun.clients.cloud.payments import (
    CheckoutCallback,
    PaymentAdapter,
    wait_for_payment,
)
from mobilerun.config.settings import get_settings
from mobilerun.core.extensions import action
from mobilerun.core.mailbox.base import Mailbox
from mobilerun.errors import PaymentRequired
from mobilerun.models.billing import Billing, ResourcePlan
from mobilerun.models.checkout import PaymentSnapshot
from mobilerun.models.context import ContextItem
from mobilerun.utils.codes import wait_for_code
from mobilerun.utils.timestamps import to_api_timestamp


class CloudMailbox(Mailbox):
    def __init__(
        self,
        mailbox_id: str | None = None,
        *,
        billing: Billing | str | None = None,
        plan: ResourcePlan | str | None = None,
        label: str | None = None,
        keep: bool = False,
        on_checkout: CheckoutCallback | None = None,
        checkout: CheckoutMode = "error",
        payment_timeout: float = 600,
        ready_timeout: float = 60,
        name: str | None = None,
        **connection: Any,
    ) -> None:
        super().__init__(name=name)
        unknown = connection.keys() - {"client", "api_key", "base_url"}
        if unknown:
            raise TypeError(
                f"Unexpected connection options: {', '.join(sorted(unknown))}"
            )
        if mailbox_id == "":
            raise ValueError("Resource id must not be empty")
        self._id = mailbox_id
        self._billing = Billing(
            get_settings().MOBILERUN_BILLING if billing is None else billing
        )
        if plan is not None:
            ResourcePlan(plan)
        self._plan = plan
        self._connection = connection
        self._label = label
        self._keep = keep
        self._on_checkout = checkout_callback(checkout, on_checkout)
        self._payment_timeout = payment_timeout
        self._ready_timeout = ready_timeout
        self._owned = False
        self._address = None
        self._lock = asyncio.Lock()

    @property
    def id(self) -> str:
        if self._id is None:
            raise RuntimeError("Await CloudMailbox(...) before using it")
        return self._id

    @property
    def address(self) -> str | None:
        return self._address

    def __await__(self) -> Generator[Any, None, Self]:
        return self._ready().__await__()

    async def _ready(self) -> Self:
        async with self._lock:
            if self._id is None:
                params: dict[str, Any] = (
                    {"label": self._label} if self._label is not None else {}
                )
                try:
                    response = await mailboxes.create(
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
                        **params,
                    )
                except PaymentRequired as exc:
                    self._id = exc.checkout.resource_id
                    self._owned = True
                    raise PaymentRequired(exc.checkout, self.wait_until_paid) from exc
                self._id = response.data.id
                self._owned = True
                if self._on_checkout is not None and response.data.status in (
                    "awaiting_payment",
                    "pending_payment",
                ):
                    response = await mailboxes.retrieve(
                        self.id, billing=self._billing, **self._connection
                    )
            else:
                response = await mailboxes.retrieve(
                    self.id, billing=self._billing, **self._connection
                )
            self._address = response.data.address
        if response.data.status in ("awaiting_payment", "pending_payment"):
            await wait_for_payment(
                PaymentAdapter("mailbox", self.id, self._payment_state, self.release),
                PaymentSnapshot(
                    response.data.status,
                    response.data.checkout_url,
                    response.data.checkout_expires_at,
                ),
                resume=self.wait_until_paid,
                on_checkout=self._on_checkout,
                timeout=self._payment_timeout,
            )
        if not (response.data.status == "active" and self._address):
            await self._wait_ready()
        return self

    async def _wait_ready(self, *, timeout: float | None = None) -> None:
        """Wait for a usable address; leave a timed-out paid mailbox intact."""
        try:
            async with asyncio.timeout(
                self._ready_timeout if timeout is None else timeout
            ):
                while True:
                    data = (
                        await mailboxes.retrieve(
                            self.id, billing=self._billing, **self._connection
                        )
                    ).data
                    self._address = data.address
                    if data.status == "active" and data.address:
                        return
                    if data.status not in (
                        "active",
                        "provisioning",
                        "awaiting_payment",
                        "pending_payment",
                    ):
                        raise RuntimeError(
                            f"Mailbox {self.id} cannot become ready (status: {data.status})"
                        )
                    await asyncio.sleep(1)
        except TimeoutError:
            raise TimeoutError(
                f"Timed out waiting for mailbox {self.id} to become active with an address; check or release it in the dashboard."
            ) from None

    async def _payment_state(self) -> PaymentSnapshot:
        data = (
            await mailboxes.retrieve(self.id, billing=self._billing, **self._connection)
        ).data
        self._address = data.address
        return PaymentSnapshot(data.status, data.checkout_url, data.checkout_expires_at)

    async def wait_until_paid(self, timeout: float = 600) -> Self:
        await self._wait_ready(timeout=timeout)
        return self

    async def __aenter__(self) -> Self:
        return await self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    @classmethod
    async def included_available(
        cls, *, billing: Billing | str | None = None, **connection: Any
    ) -> int:
        """Return a capacity snapshot, not a reservation."""
        billing = Billing(
            get_settings().MOBILERUN_BILLING if billing is None else billing
        )
        data = (await mailboxes.capacity(billing=billing, **connection)).data
        return 0 if data.status == "not_included" else data.remaining

    @classmethod
    async def list(
        cls, *, billing: Billing | str | None = None, **connection: Any
    ) -> list[Self]:
        billing = Billing(
            get_settings().MOBILERUN_BILLING if billing is None else billing
        )
        result = []
        for item in await mailboxes.list_mailboxes(billing=billing, **connection):
            mailbox = cls(item.id, billing=billing, **connection)
            mailbox._address = item.address
            result.append(mailbox)
        return result

    @action()
    async def wait_for_code(
        self,
        *,
        sender: str | None = None,
        timeout: float = 60,
        after: datetime | None = None,
    ) -> str:
        """Wait for a verification code sent to this mailbox."""
        params = {
            "after": to_api_timestamp(after or datetime.now(timezone.utc)),
            "min_length": 4,
            "max_length": 8,
        }

        async def fetch(sender: str | None) -> str | None:
            query = {**params, **({"sender": sender} if sender is not None else {})}
            response = await mailboxes.otp(
                self.id, billing=self._billing, **self._connection, **query
            )
            return response.data.code if response and response.data else None

        return await wait_for_code(fetch, timeout=timeout, sender=sender)

    async def context(self) -> tuple[ContextItem, ...]:
        return (ContextItem(self.name, "mailbox", {"address": self.address}),)

    async def release(self) -> Any:
        await self.unbind()
        result = await mailboxes.delete(
            self.id, billing=self._billing, **self._connection
        )
        self._owned = False
        return result

    async def on_close(self) -> None:
        if self._owned and not self._keep:
            await mailboxes.delete(self.id, billing=self._billing, **self._connection)
            self._owned = False
