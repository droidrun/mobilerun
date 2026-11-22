"""Shared explicit commerce option for cloud commands."""

from collections.abc import Callable
from typing import Any, TypeVar

import click

from mobilerun.config.settings import get_settings
from mobilerun.models.billing import Billing

F = TypeVar("F", bound=Callable[..., Any])


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


def creation_billing_option(function: F) -> F:
    return click.option(
        "--billing",
        type=click.Choice(["included", "paid"]),
        default=lambda: get_settings().MOBILERUN_BILLING.value,
        show_default="configured billing",
        help="included = included quota only, never charges; paid = included first, then charges.",
    )(function)


def creation_warning(billing: str) -> str:
    if billing == "included":
        return "Uses included quota only (no charges); fails if none is left."
    return "This may incur charges."
