"""Endpoint runs: fire, resolve, wait and logs."""

import asyncio
import copy
import json
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from mobilerun.models.billing import Billing

from . import clients
from .billing import require_paid
from .documents import action_type, endpoint_schema, has_ref
from .errors import SandboxDeploymentError, SandboxRunError, SandboxRunTimeout
from .flows import (
    activation_document,
    find_remote,
    flow_version,
    owned_remote,
    remote_request,
    valid_id,
)
from .models import JSON, resolve_billing
from .receipts import ReceiptStore, current_scope
from .settings import get_sandbox_settings

T = TypeVar("T")


def _validate_payload(payload: JSON, schema: Any) -> None:
    try:
        # JSON serialization rejects non-JSON payloads before transport, including NaN.
        json.dumps(payload, allow_nan=False)
        if not isinstance(payload, dict):
            raise ValueError
        if schema is not None:
            from jsonschema.validators import validator_for

            # Never resolve remote/local references as part of validation.
            if not isinstance(schema, (dict, bool)) or has_ref(schema):
                raise ValueError
            validator = validator_for(schema)
            validator.check_schema(schema)
            # Format assertions belong to the server; no optional format libraries.
            validator(schema).validate(payload)
    except Exception:
        raise SandboxRunError(
            "Sandbox payload does not match the input schema"
        ) from None


async def fire_remote(
    name: str, payload: JSON, *, billing: Billing | str | None = None
) -> JSON:
    """Fire an owned endpoint once, without waiting for execution creation."""
    mode = resolve_billing(billing)
    require_paid(billing=mode)
    store = ReceiptStore(current_scope(), name)
    with store.lock():
        receipt, flow = await owned_remote(store, mode, activating=True)
        document = await activation_document(store, receipt, flow, mode)
        on = document.get("on", {})
        if "schedule" in on:
            raise SandboxRunError(
                "Unsupported scheduled trigger; deploy as an endpoint to fire"
            )
        schema = endpoint_schema(document)
        if schema is None:
            raise SandboxRunError("Unsupported trigger; deploy as an endpoint to fire")
        version = await flow_version(receipt.function_id, document, mode)
        _validate_payload(payload, schema)
        if version.get("inputSchema") is not None:
            _validate_payload(payload, version["inputSchema"])
        fired = await remote_request(
            clients.fire_trigger(
                flow["triggerId"],
                copy.deepcopy(payload),
                uuid.uuid4().hex,
                billing=mode,
            )
        )
        invocation_id = _run_identifier(fired.get("invocationId"))
        count = fired.get("enqueuedCount")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise SandboxRunError(
                "Sandbox invocation did not enqueue an execution",
                invocation_id=invocation_id,
            )
        return {"invocation_id": invocation_id}


async def recent_runs(
    name: str, limit: int = 10, *, billing: Billing | str | None = None
) -> list[JSON]:
    """List newest executions using the boundary's ordered, paginated query."""
    if limit <= 0:
        return []
    mode = resolve_billing(billing)
    _, flow = await find_remote(name, mode)
    if flow is None:
        return []
    records = await remote_request(
        clients.list_executions(flow["id"], limit=limit, billing=mode)
    )
    return [_snapshot(record) for record in records[:limit]]


def _snapshot(record: JSON) -> JSON:
    """Keep the execution contract, not raw server errors or signed artifacts."""
    result: JSON = {
        "id": _run_identifier(record.get("id")),
        "status": record.get("status"),
    }
    for key in ("flowId", "invocationId", "startedAt", "finishedAt"):
        if isinstance(record.get(key), str):
            result[key] = record[key]
    # Use this execution's own progress, not the current (possibly redeployed) flow.
    for source in ("progress", "result"):
        metadata = record.get(source)
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("steps"), list
        ):
            continue
        if source == "result":
            result["result"] = {"steps": []}
        for step in metadata["steps"]:
            if not isinstance(step, dict):
                continue
            uses = action_type(step)
            status = step.get("status")
            body = step.get("body")
            if source == "result" and uses == "functions.run":
                function: JSON = {"uses": uses}
                if status is not None:
                    function["status"] = status if status == "success" else "failed"
                if isinstance(body, dict):
                    function["body"] = {
                        k: copy.deepcopy(body[k])
                        for k in (
                            "result",
                            "outcome",
                            "ok",
                            "partialReason",
                            "state",
                            "notes",
                        )
                        if k in body
                    }
                    if body.get("error") is not None:
                        function["status"] = "failed"
                result["result"]["steps"].append(function)
            if uses != "webhooks.send":
                continue
            if isinstance(body, dict) and body.get("error") is not None:
                status = "failed"
            elif status is None and isinstance(body, dict) and body.get("eventId"):
                status = "succeeded"
            if status == "success":
                status = "succeeded"
            if status in ("succeeded", "failed", "skipped", "cancelled"):
                result["webhookStatus"] = status
            elif status in ("pending", "running") and result.get("webhookStatus") in (
                None,
                "pending",
            ):
                result["webhookStatus"] = status
            if source == "result":
                safe_step: JSON = {"uses": "webhooks.send"}
                if "webhookStatus" in result:
                    safe_step["status"] = result["webhookStatus"]
                if isinstance(body, dict) and isinstance(body.get("eventId"), str):
                    safe_step["body"] = {"eventId": body["eventId"]}
                result["result"]["steps"].append(safe_step)
    return result


async def execution_snapshot(
    execution_id: str, *, billing: Billing | str | None = None
) -> JSON:
    """Retrieve a fresh execution snapshot, preserving partial result metadata."""
    return _snapshot(await _execution_record(execution_id, billing))


async def _execution_record(execution_id: str, billing: Billing | str | None) -> JSON:
    execution_id = _run_identifier(execution_id)
    record = await remote_request(
        clients.get_execution(execution_id, billing=resolve_billing(billing))
    )
    if record.get("id") != execution_id:
        raise SandboxRunError(
            "Sandbox execution ID mismatch", execution_id=execution_id
        )
    return record


def _timeout(invocation_id: str | None, execution_id: str | None) -> SandboxRunTimeout:
    return SandboxRunTimeout(
        "Sandbox execution deadline exceeded",
        invocation_id=invocation_id,
        execution_id=execution_id,
    )


def _run_identifier(value: Any) -> str:
    try:
        return valid_id(value)
    except SandboxDeploymentError:
        raise SandboxRunError("Invalid sandbox run identifier") from None


async def _within_deadline(
    operation: Callable[[], Awaitable[T]],
    deadline: float,
    invocation_id: str | None,
    execution_id: str | None,
) -> T:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _timeout(invocation_id, execution_id)
    try:
        result = await asyncio.wait_for(operation(), remaining)
        if time.monotonic() >= deadline:
            raise _timeout(invocation_id, execution_id)
        return result
    except TimeoutError:
        raise _timeout(invocation_id, execution_id) from None
    except SandboxRunError as error:
        error.invocation_id = error.invocation_id or invocation_id
        error.execution_id = error.execution_id or execution_id
        raise


async def _poll_pause(
    deadline: float, invocation_id: str | None, execution_id: str | None
) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _timeout(invocation_id, execution_id)
    await asyncio.sleep(
        min(get_sandbox_settings().MOBILERUN_SANDBOX_POLL_INTERVAL, remaining)
    )


async def resolve_execution(
    name: str,
    invocation_id: str,
    *,
    deadline: float,
    billing: Billing | str | None = None,
) -> str:
    """Resolve an invocation, sharing the caller's absolute monotonic deadline."""
    invocation_id = _run_identifier(invocation_id)
    if (
        not isinstance(deadline, (int, float))
        or isinstance(deadline, bool)
        or not math.isfinite(deadline)
    ):
        raise SandboxRunError(
            "Sandbox deadline must be finite", invocation_id=invocation_id
        )
    mode = resolve_billing(billing)
    while True:
        _, flow = await _within_deadline(
            lambda: find_remote(name, mode), deadline, invocation_id, None
        )
        if flow is None:
            raise SandboxRunError(
                "No flow; deploy as an endpoint first", invocation_id=invocation_id
            )
        rows = await _within_deadline(
            lambda flow_id=flow["id"]: remote_request(
                clients.list_executions(
                    flow_id, invocation_id=invocation_id, billing=mode
                )
            ),
            deadline,
            invocation_id,
            None,
        )
        for row in rows:
            if (
                row.get("invocationId", invocation_id) == invocation_id
                and row.get("flowId", flow["id"]) == flow["id"]
            ):
                return _run_identifier(row.get("id"))
        await _poll_pause(deadline, invocation_id, None)


def _function_steps(snapshot: JSON) -> list[JSON]:
    result = snapshot.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("steps"), list):
        return []
    return [
        s
        for s in result["steps"]
        if isinstance(s, dict) and action_type(s) == "functions.run"
    ]


def _check_success(snapshot: JSON, invocation_id: str | None) -> None:
    steps = _function_steps(snapshot)
    execution_id = snapshot["id"]
    if not steps:
        raise SandboxRunError(
            "Sandbox execution has no function result",
            invocation_id=invocation_id,
            execution_id=execution_id,
        )
    for step in steps:
        body = step.get("body")
        if (
            step.get("status") not in (None, "success")
            or not isinstance(body, dict)
            or body.get("ok") is False
            or body.get("error") is not None
            or body.get("outcome") not in (None, "complete", "succeeded", "partial")
            or "result" not in body
        ):
            raise SandboxRunError(
                "Sandbox function body failed or is invalid",
                invocation_id=invocation_id,
                execution_id=execution_id,
            )


async def wait_result(
    name: str,
    *,
    invocation_id: str | None = None,
    execution_id: str | None = None,
    timeout: float = 300.0,
    billing: Billing | str | None = None,
) -> JSON:
    """Wait with one deadline; a timeout does not imply remote cancellation."""
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or timeout < 0
    ):
        raise SandboxRunError("Sandbox timeout must be a finite nonnegative number")
    if invocation_id is None and execution_id is None:
        raise SandboxRunError("Provide an invocation or execution ID")
    if invocation_id is not None:
        _run_identifier(invocation_id)
    if execution_id is not None:
        _run_identifier(execution_id)
    deadline = time.monotonic() + timeout
    if execution_id is None:
        execution_id = await resolve_execution(
            name, invocation_id, deadline=deadline, billing=billing
        )
    while True:
        snapshot = await _within_deadline(
            lambda: execution_snapshot(execution_id, billing=billing),
            deadline,
            invocation_id,
            execution_id,
        )
        if invocation_id is not None and snapshot.get("invocationId") not in (
            None,
            invocation_id,
        ):
            raise SandboxRunError(
                "Sandbox invocation ID mismatch",
                invocation_id=invocation_id,
                execution_id=execution_id,
            )
        status = snapshot.get("status")
        webhook = snapshot.get("webhookStatus")
        webhook_failed = webhook in ("failed", "skipped", "cancelled")
        if status == "success" or (
            status in ("failed", "cancelled") and webhook_failed
        ):
            _check_success(snapshot, invocation_id)
            if webhook_failed:
                logging.getLogger(__name__).warning(
                    "Sandbox webhook step %s; function result is available", webhook
                )
            return snapshot
        if status not in ("pending", "running"):
            label = (
                status
                if status in ("failed", "cancelled", "skipped", "invalid", "unknown")
                else "unknown"
            )
            raise SandboxRunError(
                "Sandbox execution " + label,
                invocation_id=invocation_id,
                execution_id=execution_id,
            )
        await _poll_pause(deadline, invocation_id, execution_id)


async def execution_logs(
    execution_id: str, *, billing: Billing | str | None = None
) -> str:
    """Read inline function logs, or download available compute logs safely."""
    record = await _execution_record(execution_id, billing)
    for step in _function_steps(record):
        body = step.get("body")
        if not isinstance(body, dict):
            continue
        tail = "\n".join(
            body[key]
            for key in ("stdout", "stderr")
            if isinstance(body.get(key), str) and body[key]
        )
        if tail:
            return tail
        logs = body.get("logs")
        if logs == "available":
            if not body.get("runId"):
                raise SandboxRunError(
                    "Sandbox compute logs have no run ID", execution_id=execution_id
                )
            return await remote_request(
                clients.run_logs(
                    _run_identifier(body["runId"]), billing=resolve_billing(billing)
                )
            )
        if isinstance(logs, str) and logs != "available":
            return logs
        if isinstance(logs, list):
            return "\n".join(
                part if isinstance(part, str) else json.dumps(part, ensure_ascii=True)
                for part in logs
            )
        return ""
    raise SandboxRunError(
        "Sandbox execution has no function logs", execution_id=execution_id
    )
