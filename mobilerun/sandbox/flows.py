"""Remote flow lookup, ownership checks and activation."""

from collections.abc import Awaitable
from typing import Any, TypeVar

from mobilerun.core.webhooks import DEFAULT_WEBHOOK_EVENT, CloudWebhooks
from mobilerun.errors.cloud import CloudResourceNotFound
from mobilerun.models.billing import Billing

from . import clients
from .billing import require_paid
from .documents import (
    action_type,
    endpoint_schema,
    equivalent_document,
    has_authored_steps,
)
from .errors import SandboxDeploymentConflict, SandboxDeploymentError, SandboxRunError
from .models import (
    FILE_UPLOAD_USES,
    JSON,
    WEBHOOK_REGISTER_HINT,
    OwnershipReceipt,
    resolve_billing,
    webhook_endpoint_id,
)
from .receipts import (
    ReceiptStore,
    checkpoint,
    current_scope,
    pending_cutover,
    pending_initial,
    read_authored_document,
    stage_key,
)
from .settings import WEBHOOK_STEP_ID, get_sandbox_settings


class WebhookBindingError(SandboxDeploymentError):
    """Value-safe explanation of why the endpoint cannot be used."""


async def check_webhook(
    endpoint_id: str | None, receipt: OwnershipReceipt, mode: Billing
) -> None:
    """Verify that a webhook endpoint exists, is active and receives workflow events."""
    if endpoint_id is None:
        return
    endpoint_id = webhook_endpoint_id(endpoint_id)
    try:
        endpoint = await CloudWebhooks(billing=mode).get(endpoint_id)
    except CloudResourceNotFound:
        raise WebhookBindingError(
            f"Webhook endpoint {endpoint_id} not found. Register one: {WEBHOOK_REGISTER_HINT}",
            receipt=receipt,
        ) from None
    if endpoint.state != "ACTIVE" or endpoint.health == "blocked":
        raise WebhookBindingError(
            f"Webhook endpoint is disabled, deleted or blocked. Register one: {WEBHOOK_REGISTER_HINT}",
            receipt=receipt,
        )
    if endpoint.event_types and DEFAULT_WEBHOOK_EVENT not in endpoint.event_types:
        raise WebhookBindingError(
            f"Webhook endpoint filter excludes {DEFAULT_WEBHOOK_EVENT}. Register one: {WEBHOOK_REGISTER_HINT}",
            receipt=receipt,
        )


def valid_id(value: Any) -> str:
    """Return a cloud resource ID, rejecting values that are not plain ID strings."""
    if (
        not isinstance(value, str)
        or not value
        or any(not (c.isascii() and (c.isalnum() or c in "-_")) for c in value)
    ):
        raise SandboxDeploymentError("Invalid sandbox resource ID")
    return value


def valid_version(value: Any) -> int:
    """Return a function version, rejecting values that are not positive integers."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise SandboxDeploymentError("Invalid sandbox version")
    return value


async def delete_verified(operation: Awaitable[Any]) -> None:
    """A typed server 404 confirms that a receipt-proven resource is absent."""
    try:
        await operation
    except CloudResourceNotFound:
        pass


async def get_flow(flow_id: str, billing: Billing) -> JSON:
    """Return the one flow with this ID, or fail when its state is uncertain."""
    matches = [
        f for f in await clients.list_flows(billing=billing) if f.get("id") == flow_id
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("enabled"), bool):
        raise SandboxDeploymentConflict("Sandbox flow state is uncertain")
    return matches[0]


async def require_unshared(receipt: OwnershipReceipt, billing: Billing) -> None:
    """Fail when another flow uses the receipt's trigger or actions."""
    for flow in await clients.list_flows(billing=billing):
        if flow.get("id") == receipt.flow_id:
            if flow.get("triggerId") != receipt.trigger_id:
                raise SandboxDeploymentConflict(
                    "Sandbox trigger binding changed", receipt=receipt
                )
            continue
        if receipt.trigger_id and flow.get("triggerId") == receipt.trigger_id:
            raise SandboxDeploymentConflict(
                "Sandbox trigger is shared", receipt=receipt
            )
        bindings = await clients.list_actions(valid_id(flow.get("id")), billing=billing)
        if any(a.get("actionId") in receipt.action_ids for a in bindings):
            raise SandboxDeploymentConflict("Sandbox action is shared", receipt=receipt)


T = TypeVar("T")


async def remote_request(request: Awaitable[T]) -> T:
    """Never propagate transport bodies, credentials, or signed URLs in errors."""
    try:
        return await request
    except (SandboxDeploymentError, SandboxRunError):
        raise
    except Exception:
        raise SandboxRunError("Sandbox cloud request failed") from None


async def list_deployments(*, billing: Billing | str | None = None) -> list[JSON]:
    """List functions read-only, omitting credential and inspection metadata."""
    records = await remote_request(
        clients.list_functions(billing=resolve_billing(billing))
    )
    result = []
    for record in records:
        item = {"id": valid_id(record.get("id")), "name": record.get("name")}
        latest = record.get("latestVersion")
        if isinstance(latest, dict):
            item["latestVersion"] = {
                k: latest[k] for k in ("version", "status") if k in latest
            }
        result.append(item)
    return result


async def find_remote(name: str, mode: Billing) -> tuple[JSON, JSON | None]:
    """Return the deployed function with this name and its flow, if one exists."""
    functions = [
        f for f in await list_deployments(billing=mode) if f.get("name") == name
    ]
    if len(functions) != 1:
        raise SandboxDeploymentError(
            "Sandbox function missing or ambiguous; deploy the function first",
            missing_deployment=not functions,
        )
    flows = [
        f
        for f in await remote_request(clients.list_flows(billing=mode))
        if f.get("name") == name + get_sandbox_settings().MOBILERUN_SANDBOX_FLOW_SUFFIX
    ]
    if len(flows) > 1:
        raise SandboxDeploymentConflict("Multiple sandbox flows match")
    flow = flows[0] if flows else None
    if flow is not None:
        valid_id(flow.get("id"))
        valid_id(flow.get("triggerId"))
    return functions[0], flow


async def _latest_ready(function_id: str, mode: Billing) -> int:
    versions = await remote_request(clients.list_versions(function_id, billing=mode))
    ready = [
        valid_version(v.get("version"))
        for v in versions
        if isinstance(v, dict) and v.get("status") == "ready"
    ]
    if not ready:
        raise SandboxDeploymentError(
            "Sandbox function has no ready version; deploy a ready version first",
            missing_deployment=True,
        )
    return max(ready)


async def describe_remote(name: str, *, billing: Billing | str | None = None) -> JSON:
    """Describe current server state; neither names nor reads confer ownership."""
    mode = resolve_billing(billing)
    function, flow = await find_remote(name, mode)
    function_id = function["id"]
    result: JSON = {
        "function_id": function_id,
        "version": await _latest_ready(function_id, mode),
        "flow_id": flow["id"] if flow else None,
    }
    if flow:
        document = await remote_request(clients.get_document(flow["id"], billing=mode))
        if endpoint_schema(document) is not None:
            trigger_id = flow["triggerId"]
            result["endpoint"] = {
                "url": current_scope().origin.rstrip("/")
                + "/triggers/"
                + trigger_id
                + "/fire",
                "trigger_id": trigger_id,
            }
    return result


async def owned_remote(
    store: ReceiptStore, mode: Billing, *, activating: bool
) -> tuple[OwnershipReceipt, JSON]:
    """Return the receipt and flow after proving the receipt owns the remote flow."""
    receipt = store.read()
    function, flow = await find_remote(store.name, mode)
    if flow is None:
        raise SandboxRunError(
            "No flow; deploy as an endpoint (default) or with a schedule first",
            missing_deployment=True,
        )
    if (
        receipt is None
        or receipt.function_id != function["id"]
        or receipt.flow_id != flow["id"]
        or receipt.trigger_id != flow["triggerId"]
    ):
        raise SandboxDeploymentConflict(
            "Sandbox flow is not receipt-owned", receipt=receipt
        )
    if activating and (pending_cutover(receipt) or pending_initial(receipt)):
        raise SandboxDeploymentConflict(
            "Sandbox cutover is incomplete; finish deployment before activation",
            receipt=receipt,
        )
    bindings = tuple(
        valid_id(a.get("actionId"))
        for a in await remote_request(clients.list_actions(flow["id"], billing=mode))
    )
    if (
        not bindings
        or (set(bindings) - set(receipt.action_ids))
        or (not pending_cutover(receipt) and bindings != receipt.action_ids)
    ):
        raise SandboxDeploymentConflict(
            "Sandbox action bindings changed", receipt=receipt
        )
    await remote_request(require_unshared(receipt, mode))
    return receipt, flow


async def activation_document(
    store: ReceiptStore, receipt: OwnershipReceipt, flow: JSON, mode: Billing
) -> JSON:
    """Return the live flow document after checking it still matches what was deployed."""
    document = await remote_request(clients.get_document(flow["id"], billing=mode))
    live_steps = document.get("steps")
    has_webhook = isinstance(live_steps, list) and any(
        action_type(step) == "webhooks.send"
        for step in live_steps
        if isinstance(step, dict)
    )
    has_yaml = store.yaml_path.exists()
    authored = read_authored_document(store, receipt)
    if not has_yaml and authored is None and not has_webhook:
        return document
    if has_webhook and not has_authored_steps(authored):
        raise SandboxDeploymentConflict(
            "Missing authored webhook flow evidence; redeploy before activation",
            receipt=receipt,
        )
    if not isinstance(authored, dict) or not equivalent_document(document, authored):
        raise SandboxDeploymentConflict(
            "Sandbox remote document changed; redeploy before activation",
            receipt=receipt,
        )
    return document


def pinned_version(
    function_id: str, document: JSON, *, for_redeploy: bool = False
) -> int:
    """Return the function version a flow document pins, validating its step layout."""

    def is_webhook(step: Any) -> bool:
        return (
            isinstance(step, dict)
            and step.get("uses") == "webhooks.send"
            and step.get("id") == WEBHOOK_STEP_ID
        )

    steps = document.get("steps", [])
    optional_steps = steps[1:] if isinstance(steps, list) else []
    if optional_steps and is_webhook(optional_steps[-1]):
        optional_steps = optional_steps[:-1]
    elif optional_steps and is_webhook(optional_steps[0]):
        if not for_redeploy:
            raise SandboxDeploymentConflict(
                "Sandbox webhook precedes file uploads; redeploy before activation"
            )
        optional_steps = optional_steps[1:]
    if (
        not isinstance(steps, list)
        or not steps
        or not isinstance(steps[0], dict)
        or steps[0].get("uses") != "functions.run"
        or any(
            not isinstance(step, dict)
            or step.get("uses") != FILE_UPLOAD_USES
            or step.get("id") != f"upload_{index}"
            for index, step in enumerate(optional_steps, 1)
        )
    ):
        raise SandboxDeploymentConflict(
            "Sandbox flow must contain its owned function step"
        )
    params = steps[0].get("with")
    if not isinstance(params, dict) or params.get("functionId") != function_id:
        raise SandboxDeploymentConflict("Sandbox flow function binding changed")
    return valid_version(params.get("version"))


async def flow_version(function_id: str, document: JSON, mode: Billing) -> JSON:
    """Return the ready function version record that a flow document pins."""
    version = pinned_version(function_id, document)
    record = await remote_request(
        clients.get_version(function_id, version, billing=mode)
    )
    if record.get("version") != version or record.get("status") != "ready":
        raise SandboxDeploymentError("Sandbox flow function version is not ready")
    return record


async def check_activation_webhook(
    document: JSON, receipt: OwnershipReceipt, mode: Billing
) -> None:
    """Verify every webhook endpoint a flow document sends to."""
    for step in document["steps"]:
        if step.get("uses") == "webhooks.send":
            await check_webhook(
                valid_id((step.get("with") or {}).get("endpointId")), receipt, mode
            )


async def set_enabled(
    name: str, enabled: bool, *, billing: Billing | str | None = None
) -> JSON:
    """Activate only owned ready flows; owned shutdown needs no commerce."""
    if not isinstance(enabled, bool):
        raise SandboxDeploymentError("Sandbox enabled must be a boolean")
    mode = resolve_billing(billing)
    if enabled:
        require_paid(billing=mode)
    store = ReceiptStore(current_scope(), name)
    with store.lock():
        receipt, flow = await owned_remote(store, mode, activating=enabled)
        if enabled:
            document = await activation_document(store, receipt, flow, mode)
            await flow_version(receipt.function_id, document, mode)
            await check_activation_webhook(document, receipt, mode)
        key = stage_key(
            store.scope,
            name,
            receipt.digest or "",
            "flow:enable" if enabled else "flow:disable",
        )
        if not enabled:
            pending = pending_cutover(receipt) or pending_initial(receipt)
            if pending:
                pending["enabled"] = False
                receipt = checkpoint(store, receipt, pending)
        store.record_stage(receipt, "flow:enable" if enabled else "flow:disable", key)
        operation = clients.enable_flow if enabled else clients.disable_flow
        await remote_request(operation(flow["id"], billing=mode))
        actual = await remote_request(get_flow(flow["id"], mode))
        if actual["enabled"] is not enabled:
            raise SandboxDeploymentError(
                "Sandbox enabled state is unconfirmed", receipt=store.read()
            )
        return {"id": flow["id"], "enabled": actual["enabled"]}
