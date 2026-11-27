"""Billing checks shared by sandbox deploy, run and activation."""

from mobilerun.clients.cloud.billing import requires
from mobilerun.models.billing import Billing


@requires(Billing.PAID, operation="Sandbox deployment, endpoint run, or activation")
def require_paid(*, billing: Billing) -> None:
    """Check commerce authorization before creating local or remote resources."""


async def check_deploy_billing(billing: Billing) -> None:
    """Let interactive callers retry authorization without retrying inspection or deploy."""
    require_paid(billing=billing)
