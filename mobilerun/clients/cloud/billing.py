"""One opt-in policy for every billable cloud resource."""

import sys
from collections.abc import Awaitable, Callable, Mapping
from functools import wraps
from inspect import signature
from typing import Any, ParamSpec, TypeVar

from mobilerun.errors.cloud import BillingError, BillingRequired
from mobilerun.models.billing import Billing, Plan, ResourcePlan
from mobilerun.utils.terminal import can_prompt

P = ParamSpec("P")
R = TypeVar("R")


def billing_prompt_is_interactive() -> bool:
    """Return whether a billing prompt can use terminal stdin and stderr."""
    return can_prompt(sys.stderr)


async def _confirm_billing(error: BillingRequired) -> bool:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.output import create_output

    session: PromptSession[str] = PromptSession(output=create_output(stdout=sys.stderr))
    try:
        answer = await session.prompt_async(f"{error.confirmation} [y/N] ")
    except EOFError:
        return False
    return answer.strip().casefold() in {"y", "yes"}


async def with_billing_retry(
    operation: Callable[[Billing], Awaitable[R]],
    billing: Billing | str,
    *,
    yes: bool = False,
    prompt: Callable[[BillingRequired], Awaitable[bool]] = _confirm_billing,
    is_interactive: Callable[[], bool] | None = None,
) -> tuple[R, Billing]:
    """Retry one operation whose billing checks run before any side effects.

    Pass a callback accepting the mode for this attempt. Return the result and
    the effective mode. Without consent or a terminal on stdin and stderr,
    re-raise BillingRequired.
    """

    mode = Billing(billing)
    try:
        return await operation(mode), mode
    except BillingRequired as error:
        if mode is Billing.PAID or error.given is not mode or error.needed is mode:
            raise
        needed = error.needed
        if yes is not True:
            interactive = (
                billing_prompt_is_interactive()
                if is_interactive is None
                else is_interactive()
            )
            if not interactive or not await prompt(error):
                raise
    return await operation(needed), needed


def buys_connect_proxy(proxy: object) -> bool:
    """Return whether a proxy spec orders a new Connect proxy by country."""
    connect = proxy.get("connect") if isinstance(proxy, Mapping) else None
    return isinstance(connect, Mapping) and bool(connect.get("country"))


def rents_resource(arguments: Mapping[str, Any]) -> bool:
    """Return whether a number or mailbox call asks for a rented resource."""
    plan = arguments.get("plan")
    return plan is not None and ResourcePlan(plan) is ResourcePlan.RENT


def requires(
    minimum: Billing,
    *,
    operation: str | None = None,
    paid_if: Callable[[Mapping[str, Any]], bool] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Declare and enforce a boundary's minimum commerce mode before any work.

    ``paid_if`` receives the bound arguments and returns whether this call needs
    paid billing. Declare it on the function it guards: the guard never infers
    a rule from a module or function name.
    """

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        sig = signature(function)

        @wraps(function)
        def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(*args, **kwargs)
            given = bound.arguments["billing"]
            needed = minimum
            try:
                if paid_if is not None and paid_if(bound.arguments):
                    needed = Billing.PAID
                mode = Billing(given)
            except (ValueError, TypeError) as exc:
                error = BillingError(
                    operation=function.__name__,
                    code="invalid_billing",
                    required_billing=needed.value,
                    detail=given,
                )
                raise error from exc
            if list(Billing).index(mode) < list(Billing).index(needed):
                raise BillingRequired(
                    needed=needed, given=mode, operation=operation or function.__name__
                )
            bound.arguments["billing"] = mode
            return function(*bound.args, **bound.kwargs)

        return guarded

    return decorate


async def _billing_preference(
    resource: str,
    billing: Billing,
    *,
    plan: Plan | ResourcePlan | str | None = None,
) -> str:
    """Map authorized commerce to the SDK's resource-specific preferences."""
    mode = Billing(billing)
    if resource in {"numbers", "mailboxes"}:
        resource_plan = ResourcePlan.AUTO if plan is None else ResourcePlan(plan)
        if resource_plan is ResourcePlan.RENT:
            return "rent"
        if resource_plan is ResourcePlan.INCLUDED:
            return "included_only"
        return "included_only" if mode is Billing.INCLUDED else "included"
    if resource == "cloud phones":
        return Plan(
            plan or (Plan.SUBSCRIPTION if mode is Billing.INCLUDED else Plan.AUTO)
        ).value
    return "auto"


async def _validate_create_params(
    params: Mapping[str, object],
    *,
    allow_proxy: bool = False,
    allow_connect: bool = False,
) -> None:
    """Prevent extra fields from overriding commerce policy."""

    def forbidden(key: object) -> bool:
        return isinstance(key, str) and key.replace("_", "").lower().startswith(
            ("billing", "proxy", "connect")
        )

    extra = params.get("extra_body") or {}
    query = params.get("extra_query") or {}
    pending = [extra, query]
    while pending:
        value = pending.pop()
        if isinstance(value, Mapping):
            if any(forbidden(key) for key in value):
                raise BillingError(code="billing_override")
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    allowed = ({"proxy"} if allow_proxy else set()) | (
        {"connect"} if allow_connect else set()
    )
    if any(forbidden(key) and key not in allowed for key in params):
        raise BillingError(code="billing_override")
