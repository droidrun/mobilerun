"""Receipt-proven cleanup of owned sandbox resources."""

from mobilerun.models.billing import Billing

from . import clients
from .errors import SandboxDeploymentConflict, SandboxDeploymentError
from .flows import delete_verified, get_flow, require_unshared, valid_id
from .models import OwnershipReceipt, resolve_billing
from .receipts import (
    ReceiptStore,
    checkpoint,
    current_scope,
    pending_cutover,
    pending_initial,
    require_current_cutover_receipt,
    stage_key,
)


async def cleanup_owned(
    receipt: OwnershipReceipt,
    *,
    delete_function: bool = False,
    billing: Billing | str | None = None,
) -> None:
    """Best-effort deletion using local creation evidence, never resource names."""
    scope = current_scope()
    store = ReceiptStore(scope, receipt.name)
    mode = resolve_billing(billing)
    with store.lock():
        stored = store.read()
        if (
            stored != receipt
            or receipt.origin != store.scope.origin
            or receipt.account_id != store.scope.account_id
        ):
            raise SandboxDeploymentConflict(
                "Sandbox cleanup receipt mismatch", receipt=stored
            )
        state = pending_cutover(receipt)
        require_current_cutover_receipt(state, receipt)
        unresolved = []
        initial = pending_initial(receipt)
        unknown_initial = (
            initial
            and not initial.get("creation_resolved")
            and (
                (initial["function_pending"] and not receipt.function_id)
                or (
                    initial["flow_pending"]
                    and (
                        not receipt.flow_id
                        or not receipt.trigger_id
                        or len(receipt.action_ids) != len(initial["document"]["steps"])
                    )
                )
            )
        )
        if unknown_initial:
            # The compound create can have unreturned trigger/action/flow IDs. Deleting
            # its known dependencies would also make the original-key replay unsafe.
            raise SandboxDeploymentError(
                "Sandbox initial resource creation outcome uncertain; retry the original deployment before cleanup",
                receipt=receipt,
            )
        if state and str(state.get("pending", "")).startswith("create:"):
            # Preserve the flow and receipt so deploy can replay the original create.
            raise SandboxDeploymentError(
                "Sandbox action creation outcome uncertain; retry the original deploy before cleanup",
                receipt=receipt,
            )
        if initial:
            # Preserve resolved creation evidence before cleanup clears any IDs.
            initial["creation_resolved"] = True
            receipt = checkpoint(store, receipt, initial)

        def intent(stage: str, resource_id: str) -> None:
            nonlocal receipt
            key = stage_key(store.scope, receipt.name, resource_id, "cleanup:" + stage)
            receipt = store.record_stage(receipt, "cleanup:" + stage, key)

        flow_deleted = receipt.flow_id is None
        if receipt.flow_id:
            try:
                # A receipt-proven resource that is already absent needs no mutation.
                present = any(
                    f.get("id") == receipt.flow_id
                    for f in await clients.list_flows(billing=mode)
                )
                if not present:
                    receipt = store.record_ids(receipt, flow_id=None)
                    flow_deleted = True
                    store.remove_yaml(cleanup_confirmed=True)
                else:
                    intent("flow:disable", receipt.flow_id)
                    await require_unshared(receipt, mode)
                    bindings = {
                        a.get("actionId")
                        for a in await clients.list_actions(
                            receipt.flow_id, billing=mode
                        )
                    }
                    if not bindings <= set(receipt.action_ids):
                        raise ValueError("Foreign binding")
                    await clients.disable_flow(receipt.flow_id, billing=mode)
                    if (await get_flow(receipt.flow_id, mode))["enabled"] is not False:
                        raise ValueError("Shutdown uncertain")
                    intent("flow:delete", receipt.flow_id)
                    await delete_verified(
                        clients.delete_flow(receipt.flow_id, billing=mode)
                    )
                    if any(
                        f.get("id") == receipt.flow_id
                        for f in await clients.list_flows(billing=mode)
                    ):
                        raise ValueError("Deletion uncertain")
                    receipt = store.record_ids(receipt, flow_id=None)
                    flow_deleted = True
                    store.remove_yaml(cleanup_confirmed=True)
            except Exception:
                unresolved.append(receipt.flow_id)
        for action_id in tuple(receipt.action_ids):
            try:
                # Include the remaining owned flow: its bindings block dependency deletion.
                for flow in await clients.list_flows(billing=mode):
                    if any(
                        a.get("actionId") == action_id
                        for a in await clients.list_actions(
                            valid_id(flow.get("id")), billing=mode
                        )
                    ):
                        raise ValueError("Action still referenced")
                intent("action:delete", action_id)
                await delete_verified(
                    clients.delete_action(valid_id(action_id), billing=mode)
                )
                receipt = store.record_ids(
                    receipt,
                    action_ids=tuple(a for a in receipt.action_ids if a != action_id),
                )
            except Exception:
                unresolved.append(action_id)
        if receipt.trigger_id:
            try:
                if any(
                    f.get("triggerId") == receipt.trigger_id
                    for f in await clients.list_flows(billing=mode)
                ):
                    raise ValueError("Trigger still referenced")
                intent("trigger:delete", receipt.trigger_id)
                await delete_verified(
                    clients.delete_trigger(valid_id(receipt.trigger_id), billing=mode)
                )
                receipt = store.record_ids(receipt, trigger_id=None)
            except Exception:
                unresolved.append(receipt.trigger_id)
        if delete_function and receipt.function_id:
            try:
                if not flow_deleted or receipt.action_ids:
                    raise ValueError("Owned dependencies remain")
                # Flow bindings omit unattached actions. Inspect the complete account
                # definition inventory before deleting a potentially shared function.
                definitions = await clients.list_action_definitions(billing=mode)
                if not isinstance(definitions, list):
                    raise ValueError("Action sharing cannot be verified")
                for action in definitions:
                    if not isinstance(action, dict):
                        raise ValueError("Action definition unavailable")
                    valid_id(action.get("id"))
                    uses = action.get("uses")
                    params = action.get("params")
                    if not isinstance(uses, str) or not uses:
                        raise ValueError("Action type unavailable")
                    if params is None and uses != "functions.run":
                        continue
                    if not isinstance(params, dict) or (
                        uses == "functions.run" and "functionId" not in params
                    ):
                        raise ValueError("Action references unavailable")
                    pending = [params]
                    while pending:
                        node = pending.pop()
                        if isinstance(node, dict):
                            if (
                                "functionId" in node
                                and valid_id(node["functionId"]) == receipt.function_id
                            ):
                                raise ValueError("Function shared")
                            pending.extend(node.values())
                        elif isinstance(node, list):
                            pending.extend(node)
                # Functions may be called by any action; inspect all remaining flow documents.
                for flow in await clients.list_flows(billing=mode):
                    doc = await clients.get_document(
                        valid_id(flow.get("id")), billing=mode
                    )
                    if any(
                        s.get("with", {}).get("functionId") == receipt.function_id
                        for s in doc.get("steps", [])
                    ):
                        raise ValueError("Function shared")
                intent("function:delete", receipt.function_id)
                await delete_verified(
                    clients.delete_function(valid_id(receipt.function_id), billing=mode)
                )
                receipt = store.record_ids(receipt, function_id=None)
            except Exception:
                unresolved.append(receipt.function_id)
        if unresolved:
            raise SandboxDeploymentError(
                "Sandbox cleanup unresolved IDs: "
                + ", ".join(str(i) for i in unresolved),
                receipt=receipt,
            )
        if initial:
            initial["complete"] = True
            receipt = checkpoint(store, receipt, initial)
        if state:
            state["complete"] = True
            checkpoint(store, receipt, state)
