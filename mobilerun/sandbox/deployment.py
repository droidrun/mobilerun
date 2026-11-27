"""First deploy, redeploy and their recovery."""

import asyncio
import copy
import json
import time
from pathlib import Path
from typing import Literal

from mobilerun.models.billing import Billing

from . import clients
from .billing import require_paid
from .documents import (
    build_document,
    equivalent_document,
    has_authored_steps,
    has_ref,
    same_json,
    without_webhook,
)
from .errors import (
    FlowStagingError,
    SandboxDeclarationError,
    SandboxDeploymentConflict,
    SandboxDeploymentError,
)
from .flows import (
    WebhookBindingError,
    check_activation_webhook,
    check_webhook,
    delete_verified,
    get_flow,
    owned_remote,
    pinned_version,
    remote_request,
    require_unshared,
    valid_id,
    valid_version,
)
from .models import (
    JSON,
    Artifact,
    DeploymentReceipt,
    Endpoint,
    OwnershipReceipt,
    resolve_billing,
    validate_device,
)
from .packaging import package, verify_scheduled_artifact
from .receipts import (
    ReceiptStore,
    checkpoint,
    current_scope,
    pending_cutover,
    pending_initial,
    read_authored_document,
    require_current_cutover_receipt,
    sha256_hex,
    stage_key,
)
from .settings import get_sandbox_settings


async def _wait_ready(
    function_id: str, version: int, billing: Billing, receipt: OwnershipReceipt
) -> None:
    deadline = time.monotonic() + 120.0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SandboxDeploymentError(
                "Sandbox version inspection timed out", receipt=receipt
            )
        try:
            result = await asyncio.wait_for(
                clients.get_version(function_id, version, billing=billing), remaining
            )
        except TimeoutError:
            raise SandboxDeploymentError(
                "Sandbox version inspection timed out", receipt=receipt
            ) from None
        if time.monotonic() >= deadline:
            raise SandboxDeploymentError(
                "Sandbox version inspection timed out", receipt=receipt
            )
        if result.get("version") != version:
            raise SandboxDeploymentError(
                "Invalid sandbox version response", receipt=receipt
            )
        status = result.get("status")
        if status == "ready":
            return
        if status == "rejected":
            raise SandboxDeploymentError("Sandbox version rejected", receipt=receipt)
        if status not in {"pending", "pending_inspection", "inspecting", "processing"}:
            raise SandboxDeploymentError(
                "Invalid sandbox version status", receipt=receipt
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SandboxDeploymentError(
                "Sandbox version inspection timed out", receipt=receipt
            )
        await asyncio.sleep(
            min(get_sandbox_settings().MOBILERUN_SANDBOX_POLL_INTERVAL, remaining)
        )


async def _can_replace_initial_version(
    state: JSON,
    receipt: OwnershipReceipt,
    found: JSON | None,
    flows: list[JSON],
    mode: Billing,
) -> bool:
    """Only a rejected, receipt-owned version with no staged flow can be superseded."""
    if flows or receipt.flow_id or receipt.trigger_id or receipt.action_ids:
        return False
    if (
        state["function_pending"]
        or state["flow_pending"]
        or state["document"] is not None
    ):
        return False
    if not receipt.function_id or not found or found.get("id") != receipt.function_id:
        return False
    version = state["version"]
    if version is None or (found.get("latestVersion") or {}).get("version") != version:
        return False
    record = await clients.get_version(
        receipt.function_id, valid_version(version), billing=mode
    )
    return record.get("version") == version and record.get("status") == "rejected"


async def deploy_file(
    path: Path,
    *,
    device: str | None = None,
    device_free: bool = False,
    billing: Billing | str | None = None,
    artifact: Artifact | None = None,
) -> DeploymentReceipt:
    """Deploy a new function and optionally stage a first owned flow."""
    validate_device(device)
    mode = resolve_billing(billing)
    require_paid(billing=mode)
    if artifact is None:
        artifact = package(path)
    elif artifact.declaration.schedule is not None:
        verify_scheduled_artifact(artifact)
    decl = artifact.declaration
    if device is not None and device_free:
        raise SandboxDeclarationError("Choose a device or device-free, not both")
    selected = None if device_free else device if device is not None else decl.device
    if decl.schedule is not None and selected is None and not device_free:
        raise SandboxDeclarationError(
            "Select a device or explicitly choose device-free"
        )
    # Validate declaration before any cloud calls or local writes.
    preview_document = build_document(
        artifact,
        "placeholder",
        1,
        device=selected,
    )
    if decl.has_endpoint and has_ref(artifact.input_schema):
        raise SandboxDeclarationError("Endpoint schema contains unsupported $ref")
    scope = current_scope()
    store = ReceiptStore(scope, decl.name)
    with store.lock():
        receipt = store.read()
        initial = pending_initial(receipt) if receipt else None
        if initial and initial.get("creation_resolved"):
            raise SandboxDeploymentConflict(
                "Finish sandbox cleanup before deploying", receipt=receipt
            )
        matches = [
            f
            for f in await clients.list_functions(billing=mode)
            if f.get("name") == decl.name
        ]
        if len(matches) > 1:
            raise SandboxDeploymentConflict(
                "Multiple sandbox functions match", receipt=receipt
            )
        found = matches[0] if matches else None
        if (
            found is not None
            and (receipt is None or receipt.function_id != found.get("id"))
            and not (
                initial and initial["function_pending"] and receipt.function_id is None
            )
        ):
            raise SandboxDeploymentConflict(
                "Sandbox function is not owned", receipt=receipt
            )
        flow_name = decl.name + get_sandbox_settings().MOBILERUN_SANDBOX_FLOW_SUFFIX
        all_flows = await clients.list_flows(billing=mode)
        flows = [f for f in all_flows if f.get("name") == flow_name]
        if initial and receipt is not None:
            if "preview_document" in initial:
                matches_document = same_json(
                    initial["preview_document"], preview_document
                )
            elif (
                initial.get("document") is not None
                and receipt.function_id
                and initial.get("version")
            ):
                matches_document = same_json(
                    initial["document"],
                    build_document(
                        artifact,
                        receipt.function_id,
                        initial["version"],
                        device=selected,
                    ),
                )
            else:
                if (
                    initial["function_pending"]
                    and receipt.function_id is None
                    and not initial["flow_pending"]
                    and not (
                        receipt.flow_id or receipt.trigger_id or receipt.action_ids
                    )
                ):
                    if initial["digest"] != artifact.digest:
                        raise SandboxDeploymentConflict(
                            "Retry with the original sandbox source to recover the function ID before cleanup",
                            receipt=receipt,
                        )
                    receipt = await _create_initial_function(
                        store, receipt, artifact, initial, found, mode
                    )
                raise SandboxDeploymentConflict(
                    "Initial intent evidence is missing; cleanup is required before redeploying",
                    receipt=receipt,
                )
            if (
                initial["digest"] != artifact.digest
                or initial["device"] != selected
                or not matches_document
            ):
                if initial[
                    "digest"
                ] == artifact.digest or not await _can_replace_initial_version(
                    initial, receipt, found, flows, mode
                ):
                    raise SandboxDeploymentConflict(
                        "Finish the original initial deployment first", receipt=receipt
                    )
                # No flow staging began. Replace only this confirmed terminal version.
                initial.update(
                    digest=artifact.digest,
                    device=selected,
                    preview_document=preview_document,
                    version=None,
                    replacing_rejected=True,
                )
                receipt = checkpoint(store, receipt, initial)
            if initial.get("activation_pending"):
                receipt, flow = await owned_remote(store, mode, activating=False)
                actual = await remote_request(
                    clients.get_document(flow["id"], billing=mode)
                )
                function_id = valid_id(receipt.function_id)
                desired = build_document(
                    artifact,
                    function_id,
                    initial["version"],
                    device=selected,
                )
                if (
                    desired is None
                    or not same_json(desired, initial["document"])
                    or not equivalent_document(actual, desired)
                ):
                    raise SandboxDeploymentConflict(
                        "Initial staging intent changed", receipt=receipt
                    )
                await _wait_ready(function_id, initial["version"], mode, receipt)
                return await _finish_initial_activation(
                    store, receipt, artifact, initial, mode, flow.get("enabled")
                )
            if flows and not initial["flow_pending"]:
                raise SandboxDeploymentConflict(
                    "Sandbox flow is not owned", receipt=receipt
                )
            return await _first_deploy(
                store, receipt, artifact, selected, found, mode, flows
            )
        if receipt and receipt.flow_id:
            return await _redeploy(
                store, receipt, artifact, selected, found, all_flows, mode
            )
        if flows:
            raise SandboxDeploymentConflict(
                "Sandbox flow is not owned", receipt=receipt
            )
        if receipt is None:
            receipt = OwnershipReceipt(
                scope.origin,
                scope.account_id,
                decl.name,
                None,
                None,
                None,
                (),
                None,
                (),
            )
        return await _first_deploy(
            store, receipt, artifact, selected, found, mode, flows
        )


async def _create_initial_function(
    store: ReceiptStore,
    receipt: OwnershipReceipt,
    artifact: Artifact,
    state: JSON,
    found: JSON | None,
    mode: Billing,
) -> OwnershipReceipt:
    """Create or replay the journaled function only; never infer its ID from a name."""
    key = state["function_key"]
    receipt = store.record_stage(receipt, "function:create", key)
    try:
        result = await clients.create_function(
            receipt.name, artifact.source, key, billing=mode
        )
    except Exception:
        raise SandboxDeploymentError(
            "Sandbox function creation outcome uncertain; retry the original deployment",
            receipt=receipt,
        ) from None
    function_id = valid_id(result.get("id"))
    if receipt.function_id and receipt.function_id != function_id:
        raise SandboxDeploymentConflict(
            "Idempotent function response conflicts with remote state", receipt=receipt
        )
    receipt = store.record_ids(receipt, function_id=function_id)
    version = valid_version((result.get("latestVersion") or {}).get("version"))
    state.update(function_pending=False, version=version)
    receipt = checkpoint(store, receipt, state)
    if found and found.get("id") != function_id:
        raise SandboxDeploymentConflict(
            "Idempotent function response conflicts with remote state", receipt=receipt
        )
    return receipt


async def _first_deploy(
    store: ReceiptStore,
    receipt: OwnershipReceipt,
    artifact: Artifact,
    selected: str | None,
    found: JSON | None,
    mode: Billing,
    visible_flows: list[JSON],
) -> DeploymentReceipt:
    scope, decl = store.scope, artifact.declaration
    state = pending_initial(receipt)
    resuming = state is not None
    if state is None:
        await check_webhook(decl.webhook, receipt, mode)
        if found is None and receipt.function_id:
            raise SandboxDeploymentConflict(
                "Owned sandbox function is missing", receipt=receipt
            )
        # Completed cleanup starts a new intent, not a replay of deleted creations.
        # Journal position is persisted through the keys below before any request.
        intent_digest = sha256_hex(
            json.dumps([artifact.digest, selected, len(receipt.stages)])
        )
        state = dict(
            stage="initial",
            preview_document=build_document(
                artifact,
                "placeholder",
                1,
                device=selected,
            ),
            digest=artifact.digest,
            device=selected,
            function_pending=found is None,
            flow_pending=False,
            function_key=stage_key(scope, decl.name, intent_digest, "function:create"),
            flow_key=stage_key(scope, decl.name, intent_digest, "flow:create"),
            version=None,
            document=None,
            status="created" if found is None else "unchanged",
            complete=False,
        )
        receipt = checkpoint(store, receipt, state)
    if state["function_pending"]:
        receipt = await _create_initial_function(
            store, receipt, artifact, state, found, mode
        )
    elif state["version"] is None:
        if found is None:
            raise SandboxDeploymentConflict(
                "Owned sandbox function is missing", receipt=receipt
            )
        function_id = valid_id(found.get("id"))
        version = valid_version((found.get("latestVersion") or {}).get("version"))
        source = None
        if not state.get("replacing_rejected"):
            await _wait_ready(function_id, version, mode, receipt)
            source = await clients.get_source(function_id, version, billing=mode)
        if source == artifact.source:
            status = "unchanged"
        else:
            key = stage_key(scope, decl.name, artifact.digest, "version:create")
            receipt = store.record_stage(receipt, "version:create", key)
            result = await clients.create_version(
                function_id, artifact.source, key, billing=mode
            )
            version = valid_version(result.get("version"))
            status = "pushed"
        state.update(version=version, status=status)
        receipt = checkpoint(store, receipt, state)
    function_id = valid_id(receipt.function_id)
    version = valid_version(state["version"])
    await _wait_ready(function_id, version, mode, receipt)
    receipt = store.record_ids(receipt, digest=artifact.digest)
    if state["flow_pending"]:
        # Replay the exact staged payload before checking external dependencies.
        document = state["document"]
    else:
        document = build_document(artifact, function_id, version, device=selected)
        if (
            document is not None
            and (await clients.validate_document(document, billing=mode)).get("valid")
            is not True
        ):
            raise SandboxDeploymentError(
                "Sandbox flow validation failed", receipt=receipt
            )
    if document is not None:
        key = state["flow_key"]
        if not state["flow_pending"]:
            if resuming:
                await check_webhook(decl.webhook, receipt, mode)
            payloads = await clients.prepare_action_payloads(document, billing=mode)
            state.update(flow_pending=True, document=document, action_payloads=payloads)
            receipt = checkpoint(store, receipt, state)
        elif not isinstance(state.get("action_payloads"), list):
            raise SandboxDeploymentConflict(
                "Cannot safely replay initial staging without journaled action payloads",
                receipt=receipt,
            )
        receipt = store.record_stage(receipt, "flow:create", key)
        try:
            staged = await clients.create_document(
                document, key, billing=mode, action_payloads=state["action_payloads"]
            )
        except FlowStagingError as exc:
            receipt = store.record_ids(
                receipt,
                flow_id=exc.flow_id or receipt.flow_id,
                trigger_id=exc.trigger_id or receipt.trigger_id,
                action_ids=tuple(dict.fromkeys([*receipt.action_ids, *exc.action_ids])),
            )
            raise SandboxDeploymentError(
                "Sandbox flow staging failed", receipt=receipt
            ) from None
        except Exception:
            raise SandboxDeploymentError(
                "Sandbox flow creation outcome uncertain; retry the original deployment",
                receipt=receipt,
            ) from None
        if not isinstance(staged, dict):
            raise SandboxDeploymentError(
                "Invalid sandbox flow response", receipt=receipt
            )
        flow_id = valid_id(staged.get("flowId"))
        if receipt.flow_id and receipt.flow_id != flow_id:
            raise SandboxDeploymentConflict(
                "Idempotent flow response conflicts with remote state", receipt=receipt
            )
        receipt = store.record_ids(receipt, flow_id=flow_id)
        trigger_id = valid_id(staged.get("triggerId"))
        receipt = store.record_ids(receipt, trigger_id=trigger_id)
        actions = staged.get("actionIds")
        if not isinstance(actions, list) or not actions:
            raise SandboxDeploymentError(
                "Invalid sandbox flow response", receipt=receipt
            )
        action_ids: tuple[str, ...] = ()
        for action in actions:
            action_ids += (valid_id(action),)
            receipt = store.record_ids(receipt, action_ids=action_ids)
        if len(set(action_ids)) != len(document["steps"]) or len(action_ids) != len(
            document["steps"]
        ):
            raise SandboxDeploymentError(
                "Incomplete sandbox action IDs", receipt=receipt
            )
        if any(f.get("id") != flow_id for f in visible_flows):
            raise SandboxDeploymentConflict(
                "Idempotent flow response conflicts with remote state", receipt=receipt
            )
        store.write_yaml(document)
        state.update(flow_pending=False, activation_pending=True)
        receipt = checkpoint(store, receipt, state)
        return await _finish_initial_activation(
            store, receipt, artifact, state, mode, None
        )
    state["complete"] = True
    receipt = checkpoint(store, receipt, state)
    return _deployment(receipt, artifact, version, state["status"])


async def _finish_initial_activation(
    store: ReceiptStore,
    receipt: OwnershipReceipt,
    artifact: Artifact,
    state: JSON,
    mode: Billing,
    current: bool | None,
) -> DeploymentReceipt:
    """Close staging only after the flow reflects the journaled activation intent."""
    flow_id = valid_id(receipt.flow_id)
    target = state.get("enabled", True)
    if target:
        await check_activation_webhook(state["document"], receipt, mode)
    try:
        if current is not target:
            stage = "flow:enable" if target else "flow:disable"
            key = stage_key(store.scope, receipt.name, artifact.digest, stage)
            receipt = store.record_stage(receipt, stage, key)
            operation = clients.enable_flow if target else clients.disable_flow
            updated = await operation(flow_id, billing=mode)
            if not isinstance(updated, dict) or updated.get("enabled") is not target:
                raise ValueError("Activation not confirmed")
        if (await get_flow(flow_id, mode))["enabled"] is not target:
            raise ValueError("Activation not confirmed")
    except Exception:
        raise SandboxDeploymentError(
            "Sandbox flow activation failed or uncertain", receipt=receipt
        ) from None
    state["complete"] = True
    receipt = checkpoint(store, receipt, state)
    return _deployment(receipt, artifact, state["version"], state["status"])


def _deployment(
    receipt: OwnershipReceipt,
    artifact: Artifact,
    version: int,
    status: Literal["created", "pushed", "unchanged"],
    warnings: tuple[str, ...] = (),
) -> DeploymentReceipt:
    endpoint = None
    if artifact.declaration.has_endpoint:
        trigger_id = valid_id(receipt.trigger_id)
        endpoint = Endpoint(
            receipt.origin.rstrip("/") + "/triggers/" + trigger_id + "/fire",
            trigger_id,
        )
    return DeploymentReceipt(
        receipt.name,
        valid_id(receipt.function_id),
        version,
        status,
        receipt.flow_id,
        receipt.trigger_id,
        receipt.action_ids,
        artifact.digest,
        endpoint,
        warnings,
    )


async def _redeploy(
    store: ReceiptStore,
    receipt: OwnershipReceipt,
    artifact: Artifact,
    selected: str | None,
    found: JSON | None,
    all_flows: list[JSON],
    mode: Billing,
) -> DeploymentReceipt:
    """Reconcile an owned flow with an on-disk write-ahead cutover journal."""
    state = pending_cutover(receipt)
    require_current_cutover_receipt(state, receipt)
    if not found or not receipt.trigger_id or not receipt.action_ids:
        raise SandboxDeploymentConflict(
            "Sandbox ownership is incomplete", receipt=receipt
        )
    valid_id(receipt.function_id)
    flow_id = valid_id(receipt.flow_id)
    valid_id(receipt.trigger_id)
    for action_id in receipt.action_ids:
        valid_id(action_id)
    flow = await get_flow(receipt.flow_id, mode)
    expected_name = receipt.name + get_sandbox_settings().MOBILERUN_SANDBOX_FLOW_SUFFIX
    if (
        flow.get("name") != expected_name
        or sum(f.get("name") == expected_name for f in all_flows) != 1
    ):
        raise SandboxDeploymentConflict(
            "Sandbox flow ownership mismatch", receipt=receipt
        )
    await require_unshared(receipt, mode)
    bindings = tuple(
        valid_id(a.get("actionId"))
        for a in await clients.list_actions(receipt.flow_id, billing=mode)
    )
    allowed = (
        [tuple(state["old_ids"]), tuple(state["new_ids"])]
        if state
        else [receipt.action_ids]
    )
    if bindings not in allowed:
        raise SandboxDeploymentConflict(
            "Sandbox action bindings changed", receipt=receipt
        )
    actual = await clients.get_document(receipt.flow_id, billing=mode)
    if state:
        requested = build_document(
            artifact, receipt.function_id, state["version"], device=selected
        )
        desired_changed = not same_json(requested, state["desired"])
        retarget = (
            state.get("webhook_unusable")
            and desired_changed
            and requested is not None
            and same_json(without_webhook(requested), without_webhook(state["desired"]))
        )
        if (
            state["digest"] != artifact.digest
            or state["device"] != selected
            or (desired_changed and not retarget)
        ):
            raise SandboxDeploymentConflict(
                "Finish the pending sandbox deployment first", receipt=receipt
            )
        expected = copy.deepcopy(state["before"])
        # Only the stage in flight may have either its old or new remote value.
        candidates = [expected]
        for stage, field in (
            ("replace", "steps"),
            ("trigger", "on"),
            ("devices", "devices"),
        ):
            if stage in state["done"]:
                for candidate in candidates:
                    candidate[field] = state["desired"].get(field, [])
            elif state.get("pending") == stage:
                changed = copy.deepcopy(candidates[0])
                changed[field] = state["desired"].get(field, [])
                candidates.append(changed)
        delivery_candidates = []
        if "delivery" in state["done"]:
            delivery_candidates = candidates
        elif state.get("pending") == "delivery":
            delivery_candidates = [copy.deepcopy(candidates[0])]
            candidates.extend(delivery_candidates)
        for candidate in delivery_candidates:
            settings = candidate.setdefault("settings", {})
            if delivery := state["desired"].get("settings", {}).get("delivery"):
                settings["delivery"] = delivery
            else:
                settings.pop("delivery", None)
        if not any(equivalent_document(actual, c) for c in candidates):
            raise SandboxDeploymentConflict(
                "Sandbox remote state diverged", receipt=receipt
            )
        version = state["version"]
        await _wait_ready(receipt.function_id, version, mode, receipt)
        desired = state["desired"]
        if retarget and requested is not None:
            await check_activation_webhook(requested, receipt, mode)
            if (await clients.validate_document(requested, billing=mode)).get(
                "valid"
            ) is not True:
                raise SandboxDeploymentError(
                    "Sandbox flow validation failed", receipt=receipt
                )
            # The observed document is a verified cutover state, not new ownership.
            # Keep every known action until the replacement is confirmed active.
            state.update(
                before=actual,
                desired=requested,
                action_payloads=await clients.prepare_action_payloads(
                    requested, billing=mode
                ),
                old_ids=list(bindings),
                retire_ids=list(receipt.action_ids),
                new_ids=[],
                done=[],
                pending=None,
                webhook_unusable=False,
                retarget_generation=len(receipt.stages),
            )
            receipt = checkpoint(store, receipt, state)
            desired = requested
    else:
        has_yaml = store.yaml_path.exists()
        authored = read_authored_document(store, receipt)
        if has_yaml or authored is not None:
            if (
                not isinstance(authored, dict)
                or (not has_yaml and not has_authored_steps(authored))
                or not equivalent_document(actual, authored)
            ):
                raise SandboxDeploymentConflict(
                    "Sandbox remote document changed", receipt=receipt
                )
        pinned = pinned_version(receipt.function_id, actual, for_redeploy=True)
        version = pinned
        source = await clients.get_source(receipt.function_id, version, billing=mode)
        if source != artifact.source:
            version = valid_version((found.get("latestVersion") or {}).get("version"))
            if version != pinned:
                source = await clients.get_source(
                    receipt.function_id, version, billing=mode
                )
        if source == artifact.source:
            await _wait_ready(receipt.function_id, version, mode, receipt)
        await check_webhook(artifact.declaration.webhook, receipt, mode)
        desired = build_document(
            artifact,
            receipt.function_id,
            version,
            device=selected,
        )
        if desired is None:
            raise SandboxDeploymentConflict(
                "Remove the owned flow explicitly first", receipt=receipt
            )
        if source == artifact.source and equivalent_document(actual, desired):
            if artifact.declaration.webhook and not store.yaml_path.exists():
                store.write_yaml(desired)
            return _deployment(receipt, artifact, version, "unchanged")
        if authored is None:
            raise SandboxDeploymentConflict(
                "Sandbox authored document evidence is missing", receipt=receipt
            )
        if source != artifact.source:
            key = stage_key(
                store.scope,
                receipt.name,
                artifact.digest,
                f"version:create:{receipt.function_id}:{pinned}:{','.join(bindings)}",
            )
            receipt = store.record_stage(receipt, "version:create", key)
            result = await clients.create_version(
                receipt.function_id, artifact.source, key, billing=mode
            )
            version = valid_version(result.get("version"))
            await _wait_ready(receipt.function_id, version, mode, receipt)
            desired = build_document(
                artifact,
                receipt.function_id,
                version,
                device=selected,
            )
        if (await clients.validate_document(desired, billing=mode)).get(
            "valid"
        ) is not True:
            raise SandboxDeploymentError(
                "Sandbox flow validation failed", receipt=receipt
            )
        state = dict(
            stage="cutover",
            digest=artifact.digest,
            device=selected,
            version=version,
            before=actual,
            desired=desired,
            action_payloads=await clients.prepare_action_payloads(
                desired, billing=mode
            ),
            old_ids=list(receipt.action_ids),
            new_ids=[],
            enabled=flow["enabled"],
            done=[],
            pending=None,
            complete=False,
        )
        receipt = checkpoint(store, receipt, state)

    payloads = state.get("action_payloads")
    if not isinstance(payloads, list) or len(payloads) != len(desired["steps"]):
        raise SandboxDeploymentConflict(
            "Cannot safely resume cutover without journaled action payloads",
            receipt=receipt,
        )

    def save():
        nonlocal receipt
        receipt = checkpoint(store, receipt, state)

    def intent(stage):
        nonlocal receipt
        state["pending"] = stage
        save()
        # Includes the full desired document, so device/schema-only changes get new keys.
        identity = [state["desired"], state["old_ids"]]
        if "retarget_generation" in state:
            identity.append(state["retarget_generation"])
        digest = sha256_hex(json.dumps(identity, sort_keys=True))
        key = stage_key(store.scope, receipt.name, digest, stage)
        receipt = store.record_stage(receipt, stage, key)
        return key

    def done(stage):
        state["done"].append(stage)
        state["pending"] = None
        save()

    delivery = desired.get("settings", {}).get("delivery")
    delivery_changed = not same_json(
        state["before"].get("settings", {}).get("delivery"), delivery
    )

    async def update_delivery() -> None:
        if delivery_changed and "delivery" not in state["done"]:
            key = intent("delivery")
            await clients.update_flow_delivery(
                valid_id(receipt.flow_id), delivery, key, billing=mode
            )
            done("delivery")

    try:
        activated = "enable" in state["done"]
        if not activated:
            # Re-prove shutdown, including after an uncertain enable. Once enable
            # is confirmed, retries only verify the live flow and retire old actions.
            key = stage_key(store.scope, receipt.name, artifact.digest, "disable")
            receipt = store.record_stage(receipt, "disable", key)
            await clients.disable_flow(receipt.flow_id, billing=mode)
            if (await get_flow(receipt.flow_id, mode))["enabled"] is not False:
                raise ValueError("Shutdown not confirmed")
            if "disable" not in state["done"]:
                state["done"].append("disable")
                save()
        for index, payload in enumerate(payloads):
            stage = f"create:{index}"
            if stage in state["done"]:
                continue
            key = intent(stage)
            action = await clients.create_action(
                payload,
                key,
                billing=mode,
            )
            action_id = valid_id(action.get("id"))
            state["new_ids"].append(action_id)
            receipt = store.record_ids(
                receipt,
                action_ids=tuple(dict.fromkeys([*receipt.action_ids, action_id])),
            )
            done(stage)
        # The service permits uploads without delivery, but not the reverse.
        if delivery is None:
            await update_delivery()
        if "replace" not in state["done"]:
            if (await get_flow(receipt.flow_id, mode))["enabled"] is not False:
                raise ValueError("Shutdown no longer confirmed")
            await check_activation_webhook(desired, receipt, mode)
            key = intent("replace")
            refs = [
                dict(action_id=a, position=i, key=desired["steps"][i]["id"])
                for i, a in enumerate(state["new_ids"])
            ]
            await clients.replace_flow_actions(receipt.flow_id, refs, key, billing=mode)
            done("replace")
        if (
            not same_json(state["before"]["on"], desired["on"])
            and "trigger" not in state["done"]
        ):
            await require_unshared(receipt, mode)
            on = desired["on"]
            params = dict(
                activation="custom",
                custom_payload_schema=on.get("custom", {}).get("inputs"),
                schedule_rule=None,
                timezone=None,
            )
            if "schedule" in on:
                params.update(
                    activation="schedule",
                    schedule_rule=dict(type="cron", expression=on["schedule"]["cron"]),
                    timezone=on["schedule"]["timezone"],
                )
            key = intent("trigger")
            await clients.update_trigger(receipt.trigger_id, params, key, billing=mode)
            done("trigger")
        if (
            not same_json(
                state["before"].get("devices", []), desired.get("devices", [])
            )
            and "devices" not in state["done"]
        ):
            key = intent("devices")
            await clients.update_flow_devices(
                receipt.flow_id, desired.get("devices", []), key, billing=mode
            )
            done("devices")
        if delivery is not None:
            await update_delivery()
        if not activated:
            intent("verify")
        await require_unshared(receipt, mode)
        expected_enabled = state["enabled"] if activated else False
        if (await get_flow(flow_id, mode))[
            "enabled"
        ] is not expected_enabled or not equivalent_document(
            await clients.get_document(flow_id, billing=mode), desired
        ):
            raise ValueError("Cutover verification failed")
        verified = tuple(
            a.get("actionId") for a in await clients.list_actions(flow_id, billing=mode)
        )
        if verified != tuple(state["new_ids"]):
            raise ValueError("Cutover bindings changed")
        if not activated:
            done("verify")
        if state["enabled"] and not activated:
            await check_activation_webhook(desired, receipt, mode)
            intent("enable")
            await clients.enable_flow(flow_id, billing=mode)
            if (await get_flow(flow_id, mode))["enabled"] is not True:
                raise ValueError("Enable not confirmed")
            done("enable")
        unresolved = []
        retire_ids: list[str] = (
            state["retire_ids"] if "retire_ids" in state else state["old_ids"]
        )
        for action_id in retire_ids:
            stage = "delete:" + action_id
            if stage in state["done"]:
                continue
            try:
                intent(stage)
                await require_unshared(receipt, mode)
                await delete_verified(clients.delete_action(action_id, billing=mode))
                done(stage)
            except Exception:
                unresolved.append(action_id)
        if unresolved:
            raise ValueError("Old actions unresolved")
        receipt = store.record_ids(
            receipt, action_ids=tuple(state["new_ids"]), digest=artifact.digest
        )
        store.write_yaml(desired)
        state["complete"] = True
        save()
        warnings = (
            () if state["enabled"] else ("Flow remains disabled (enabled=False)",)
        )
        return _deployment(receipt, artifact, version, "pushed", warnings)
    except Exception as exc:
        # A confirmed activation must not be undone by a cleanup failure.
        try:
            restore_old = (
                isinstance(exc, WebhookBindingError)
                and state["enabled"]
                and not {"replace", "delivery"}.intersection(state["done"])
                and state.get("pending") not in ("replace", "delivery")
            )
            if restore_old:
                # Only definitions were created; no live binding or settings changed.
                await require_unshared(receipt, mode)
                current = await clients.get_document(flow_id, billing=mode)
                current_ids = [
                    row.get("actionId")
                    for row in await clients.list_actions(flow_id, billing=mode)
                ]
                if current_ids != state["old_ids"] or not equivalent_document(
                    current, state["before"]
                ):
                    raise ValueError("Previous workflow state changed")
                await check_activation_webhook(state["before"], receipt, mode)
                key = stage_key(
                    store.scope, receipt.name, artifact.digest, "restore:enable"
                )
                receipt = store.record_stage(receipt, "restore:enable", key)
                await clients.enable_flow(flow_id, billing=mode)
            elif "enable" not in state["done"]:
                key = stage_key(
                    store.scope, receipt.name, artifact.digest, "failure:shutdown"
                )
                receipt = store.record_stage(receipt, "failure:shutdown", key)
                await clients.disable_flow(receipt.flow_id, billing=mode)
        except Exception:
            pass
        try:
            enabled = (await get_flow(receipt.flow_id, mode))["enabled"]
        except Exception:
            enabled = "uncertain"
        state["warning"] = f"Cutover incomplete; flow enabled={enabled}"
        if isinstance(exc, WebhookBindingError):
            state["webhook_unusable"] = True
            state[
                "warning"
            ] += f". {exc}. Redeploy with a usable endpoint ID or remove webhook=."
        save()
        raise SandboxDeploymentError(state["warning"], receipt=receipt) from None
