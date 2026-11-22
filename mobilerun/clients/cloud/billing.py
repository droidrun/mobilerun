"""One opt-in policy for every billable cloud resource."""

from collections.abc import Callable, Mapping
from functools import wraps
from inspect import signature
from typing import ParamSpec, TypeVar

from mobilerun.errors.cloud import BillingError
from mobilerun.models.billing import Billing, Plan, ResourcePlan

P = ParamSpec("P")
R = TypeVar("R")


def requires(minimum: Billing) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Declare and enforce a boundary's minimum commerce mode before any work."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        sig = signature(function)

        @wraps(function)
        def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(*args, **kwargs)
            given = bound.arguments["billing"]
            needed = minimum
            params = bound.arguments.get("params", {})
            proxy = bound.arguments.get("proxy") or params
            connect = proxy.get("connect", {}) if isinstance(proxy, Mapping) else {}
            if function.__module__.endswith((".devices", ".proxies")):
                if isinstance(connect, Mapping) and connect.get("country"):
                    needed = Billing.PAID
            try:
                mode = Billing(given)
                if (
                    function.__module__.endswith(".devices")
                    and function.__name__ == "create"
                ):
                    plan = bound.arguments.get("plan")
                    if plan is not None:
                        plan = Plan(plan)
                    if plan in (Plan.AUTO, Plan.MINUTE):
                        needed = Billing.PAID
                if (
                    function.__module__.endswith((".numbers", ".mailboxes"))
                    and function.__name__ == "create"
                ):
                    plan = bound.arguments.get("plan")
                    if plan is not None and ResourcePlan(plan) is ResourcePlan.RENT:
                        needed = Billing.PAID
                if list(Billing).index(mode) < list(Billing).index(needed):
                    raise ValueError
            except (ValueError, TypeError) as exc:
                error = BillingError(
                    operation=function.__name__,
                    code="invalid_billing",
                    required_billing=needed.value,
                    detail=given,
                )
                raise error from exc
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
