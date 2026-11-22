"""Billing and phone-plan options for executable demos."""

import argparse

from mobilerun.config.settings import get_settings
from mobilerun.models import Billing, Plan

BILLING_HELP = "included = included quota only, never charges; paid = included first, then charges (default: configured billing)."
PLAN_HELP = "auto = subscription slot first, then per-minute; minute = always per-minute; subscription = included slot only."


def add_billing(parser: argparse.ArgumentParser, *, plan: bool = False) -> None:
    parser.add_argument(
        "--billing", choices=("included", "paid"), default=None, help=BILLING_HELP
    )
    if plan:
        parser.add_argument(
            "--plan",
            choices=("auto", "minute", "subscription"),
            default=None,
            help=PLAN_HELP,
        )


def billing_value(value: str | None) -> Billing:
    return Billing(value) if value else get_settings().MOBILERUN_BILLING


def check_phone_plan(billing: Billing, plan: str | None) -> None:
    if billing == Billing.INCLUDED and plan in (Plan.AUTO, Plan.MINUTE):
        raise ValueError(
            "--billing included requires --plan subscription (or omit --plan)"
        )


def confirmation(billing: Billing) -> str:
    if billing == Billing.INCLUDED:
        return "This uses included quota only (no charges); fails if none is left. Continue? [y/N] "
    if billing == Billing.NO_COMMERCE:
        return (
            "Configured billing no_commerce blocks resource creation. Continue? [y/N] "
        )
    return "This may use paid cloud services. Continue? [y/N] "
