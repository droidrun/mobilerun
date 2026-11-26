"""Shared explicit commerce option for cloud commands."""

import sys
from collections.abc import Awaitable, Callable
from contextlib import redirect_stdout
from typing import Any, TypeVar

import click

from mobilerun.clients.cloud.billing import (
    billing_prompt_is_interactive,
    with_billing_retry,
)
from mobilerun.config.settings import get_settings
from mobilerun.errors.base import MobilerunError
from mobilerun.errors.cloud import BillingRequired
from mobilerun.models.billing import Billing

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


class BillingConsentError(MobilerunError):
    """Consent was declined or unavailable, potentially inside an action."""

    value_safe = True


async def invoke_with_billing(
    operation: Callable[[Billing], Awaitable[T]],
    billing: Billing,
    *,
    yes: bool,
    confirm_paid: str | None = None,
) -> tuple[T, Billing]:
    """Confirm opt-in paid operations up front or once at their retry guard."""
    if confirm_paid is not None and billing is Billing.PAID:
        await confirm_paid_action(confirm_paid, yes=yes)

    async def prompt(error: BillingRequired) -> bool:
        return await _confirm(error.confirmation)

    return await with_billing_retry(
        operation,
        billing,
        yes=yes,
        prompt=prompt,
    )


async def confirm_paid_action(operation: str, *, yes: bool) -> None:
    """Ask before a paid action, or require explicit consent in automation."""
    if yes:
        return
    if not billing_prompt_is_interactive():
        raise BillingConsentError(
            "Refusing to proceed without confirmation; pass --yes."
        )
    await _confirm(f"{operation} with paid billing (charges may apply)?")


async def _confirm(message: str) -> bool:
    from mobilerun.cli.base import safety

    with redirect_stdout(sys.stderr):
        if not await safety.confirm_prompt_async(message, default=False):
            raise BillingConsentError("Cancelled.")
    return True


def billing_option(function: F) -> F:
    return click.option(
        "--billing",
        type=click.Choice([mode.value for mode in Billing]),
        default=lambda: get_settings().MOBILERUN_BILLING.value,
        show_default="MOBILERUN_BILLING",
        help=(
            "Billing mode: no_commerce blocks purchases; included allows included resources; "
            "paid allows paid resources. Default: MOBILERUN_BILLING (paid)."
        ),
    )(function)


def creation_warning(billing: str) -> str:
    if billing == "included":
        return "Uses included quota only (no charges); fails if none is left."
    return "This may incur charges."
