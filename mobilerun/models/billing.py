"""Billing modes shared by configuration and core resources."""

from enum import StrEnum


class Billing(StrEnum):
    """NO_COMMERCE blocks creation; INCLUDED uses included quota; PAID permits charges."""

    NO_COMMERCE = "no_commerce"
    INCLUDED = "included"
    PAID = "paid"


class ResourcePlan(StrEnum):
    """Phone number and mailbox allocation preferences."""

    AUTO = "auto"
    INCLUDED = "included"
    RENT = "rent"


class Plan(StrEnum):
    """Cloud phone billing plans.

    AUTO uses a subscription first, then falls back to minutes.
    MINUTE uses per-minute balance only; exhaustion raises MINUTE_BALANCE_EXHAUSTED.
    SUBSCRIPTION uses a subscription slot only, never minutes.
    """

    AUTO = "auto"
    MINUTE = "minute"
    SUBSCRIPTION = "subscription"
