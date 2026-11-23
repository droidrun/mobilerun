"""Shared, presentation-free checkout polling for cloud resources."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from mobilerun.errors import CloudAPIError, PaymentRequired
from mobilerun.models.checkout import Checkout, PaymentSnapshot

T = TypeVar("T")
CheckoutCallback = Callable[[Checkout], Awaitable[None] | None]


@dataclass(frozen=True)
class PaymentAdapter:
    resource: str
    resource_id: str
    retrieve: Callable[[], Awaitable[PaymentSnapshot]]
    cancel: Callable[[], Awaitable[object]]
    awaiting: frozenset[str] = frozenset({"awaiting_payment", "pending_payment"})
    preparing: frozenset[str] = frozenset()
    active: frozenset[str] = frozenset({"active"})
    provisioning: frozenset[str] = frozenset({"provisioning"})


async def wait_until_active(adapter: PaymentAdapter, *, timeout: float = 600) -> None:
    """Wait for payment and provisioning without changing resource ownership."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    attempt = 0
    try:
        if timeout <= 0:
            raise TimeoutError
        async with asyncio.timeout(timeout):
            while True:
                state = await adapter.retrieve()
                if state.status in adapter.active:
                    return
                if (
                    state.status
                    not in adapter.awaiting | adapter.preparing | adapter.provisioning
                ):
                    error = CloudAPIError(code="payment_failed", status=state.status)
                    raise error
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                await asyncio.sleep(min((1, 2, 3, 5)[min(attempt, 3)], remaining))
                attempt += 1
    except TimeoutError as exc:
        raise TimeoutError(
            f"Timed out waiting for {adapter.resource} {adapter.resource_id}; check the dashboard."
        ) from exc
    raise TimeoutError(
        f"Timed out waiting for {adapter.resource} {adapter.resource_id}; check the dashboard."
    )


async def wait_for_payment(
    adapter: PaymentAdapter,
    initial: PaymentSnapshot,
    *,
    resume: Callable[[float], Awaitable[T]],
    on_checkout: CheckoutCallback | None = None,
    timeout: float = 600,
    checkout_timeout: float = 30,
) -> None:
    """Find checkout and hand presentation to the caller, or expose resumable error."""
    if initial.status not in adapter.awaiting | adapter.preparing:
        return
    state = initial
    loop = asyncio.get_running_loop()
    deadline = loop.time() + checkout_timeout
    attempt = 0
    try:
        while True:
            if state.status in adapter.awaiting and state.url:
                checkout = Checkout(
                    state.url, state.expires_at, adapter.resource, adapter.resource_id
                )
                if on_checkout is None:
                    raise PaymentRequired(checkout, resume)
                async with asyncio.timeout(timeout):
                    try:
                        result = on_checkout(checkout)
                        if inspect.isawaitable(result):
                            await result
                    except Exception as exc:
                        try:
                            await adapter.cancel()
                        except Exception as cleanup_error:
                            exc.add_note(
                                f"Could not cancel {adapter.resource} {adapter.resource_id}: {cleanup_error}; check the dashboard."
                            )
                        raise
                    await wait_until_active(adapter, timeout=timeout)
                return
            if state.status not in adapter.awaiting | adapter.preparing:
                if (
                    initial.status in adapter.preparing
                    and state.status in adapter.provisioning | adapter.active
                ):
                    return
                error = CloudAPIError(code="checkout_missing", status=state.status)
                raise error
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError
            async with asyncio.timeout(remaining):
                await asyncio.sleep(min((1, 2, 3, 5)[min(attempt, 3)], remaining))
                state = await adapter.retrieve()
            attempt += 1
    except (TimeoutError, asyncio.CancelledError, KeyboardInterrupt) as exc:
        if on_checkout is not None:
            try:
                await adapter.cancel()
                exc.add_note(
                    f"Cancelled pending {adapter.resource} {adapter.resource_id}."
                )
            except Exception as cleanup_error:
                exc.add_note(
                    f"Could not cancel {adapter.resource} {adapter.resource_id}: {cleanup_error}; check the dashboard."
                )
        if isinstance(exc, TimeoutError):
            error = TimeoutError(
                f"Timed out waiting for {adapter.resource} {adapter.resource_id}; check the dashboard."
            )
            for note in exc.__notes__ if hasattr(exc, "__notes__") else ():
                error.add_note(note)
            raise error from exc
        raise
