"""Shared explicit commerce option for cloud commands."""

import sys
from collections.abc import Awaitable, Callable
from contextlib import redirect_stdout
from typing import Any, TypeVar

import click

from mobilerun.clients.cloud.billing import with_billing_retry
from mobilerun.config.settings import get_settings
from mobilerun.errors.base import MobilerunError
from mobilerun.errors.cloud import BillingRequired
from mobilerun.models.billing import Billing

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


class BillingDeclined(MobilerunError):
    """User declined a billing retry, potentially inside an extension action."""

    value_safe = True

    def __init__(self) -> None:
        super().__init__("Cancelled.")


async def invoke_with_billing(
    operation: Callable[[Billing], Awaitable[T]], billing: Billing, *, yes: bool
) -> tuple[T, Billing]:
    """Prompt centrally before retrying an operation with an audited guard."""
    from mobilerun.cli.base import safety

    async def prompt(error: BillingRequired) -> bool:
        with redirect_stdout(sys.stderr):
            accepted = await safety.confirm_prompt_async(
                error.confirmation,
                default=False,
            )
        if not accepted:
            raise BillingDeclined()
        return accepted

    return await with_billing_retry(
        operation,
        billing,
        yes=yes,
        prompt=prompt,
    )


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
