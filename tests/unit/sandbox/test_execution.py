"""Fresh remote reads, safe invocation, and one-budget execution polling."""

import asyncio
import copy
import json
import logging
import tomllib
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from mobilerun.errors.cloud import BillingError
from mobilerun.sandbox import clients
from mobilerun.sandbox import execution as sandbox_execution
from mobilerun.sandbox import flows as sandbox_flows
from mobilerun.sandbox import receipts as sandbox_receipts
from mobilerun.sandbox.errors import (
    SandboxDeploymentConflict,
    SandboxDeploymentError,
    SandboxRunError,
    SandboxRunTimeout,
)
from tests.support.paths import REPO_ROOT
from tests.support.sandbox_fixtures import execution
from tests.support.sandbox_fixtures import remote as remote_fixture

remote = remote_fixture

LIST_EXECUTIONS = clients.list_executions


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["enable", "fire"])
@pytest.mark.parametrize("stage", [None, "initial", "cutover"])
async def test_webhook_activation_without_yaml_requires_matching_evidence(
    remote, stage, operation
):
    hook = {
        "id": "webhook",
        "name": "demo · webhook",
        "uses": "webhooks.send",
        "with": {
            "endpointId": "abcdef00-0000-4000-8000-000000000001",
            "data": {"result": "{{steps.function.body.result}}"},
        },
    }
    remote.document["steps"][0].update(id="function", name="demo · function")
    remote.document["steps"].append(hook)
    receipt = replace(remote.receipt, action_ids=("action", "send"))
    remote.store.write(receipt)
    remote.fake.list_actions.return_value.append({"actionId": "send"})
    if stage:
        sandbox_receipts.checkpoint(
            remote.store,
            receipt,
            {
                "stage": stage,
                "complete": True,
                ("document" if stage == "initial" else "desired"): remote.document,
            },
        )
        if operation == "fire":
            assert await sandbox_execution.fire_remote("demo", {"n": 1}) == {
                "invocation_id": "invocation"
            }
            remote.fake.fire_trigger.reset_mock()
    assert not remote.store.yaml_path.exists()
    hook["with"]["endpointId"] = "abcdef00-0000-4000-8000-000000000002"
    with pytest.raises(SandboxDeploymentConflict, match="redeploy"):
        if operation == "enable":
            await sandbox_flows.set_enabled("demo", True)
        else:
            await sandbox_execution.fire_remote("demo", {"n": 1})
    remote.fake.enable_flow.assert_not_awaited()
    remote.fake.fire_trigger.assert_not_awaited()


@pytest.mark.anyio
async def test_read_only_discovery_requires_no_receipt_and_filters_secret_metadata(
    remote,
):
    remote.store.receipt_path.unlink()
    rows = await sandbox_flows.list_deployments(billing="no_commerce")
    assert rows[0]["name"] == "demo" and "PRIVATE" not in repr(rows)
    info = await sandbox_flows.describe_remote("demo", billing="no_commerce")
    assert info["function_id"] == "function" and info["version"] == 2
    assert info["flow_id"] == "flow"
    assert info["endpoint"] == {
        "url": "https://example.test/v1/triggers/trigger/fire",
        "trigger_id": "trigger",
    }


@pytest.mark.anyio
async def test_description_fresh_reads_latest_ready_numeric_version(remote):
    assert (await sandbox_flows.describe_remote("demo"))["version"] == 2
    remote.fake.list_versions.return_value = [{"version": 4, "status": "ready"}]
    remote.fake.list_flows.return_value = []
    info = await sandbox_flows.describe_remote("demo")
    assert info["version"] == 4 and info["flow_id"] is None
    assert not info.get("endpoint")


@pytest.mark.anyio
async def test_missing_and_duplicate_function_or_flow_rejected(remote):
    remote.fake.list_functions.return_value *= 2
    with pytest.raises(SandboxDeploymentError):
        await sandbox_flows.describe_remote("demo")
    remote.fake.list_functions.return_value = []
    with pytest.raises(SandboxDeploymentError, match="deploy"):
        await sandbox_flows.describe_remote("demo")


@pytest.mark.anyio
async def test_no_ready_version_is_not_reported_ready(remote):
    remote.fake.list_versions.return_value = [{"version": 3, "status": "pending"}]
    with pytest.raises(SandboxDeploymentError):
        await sandbox_flows.describe_remote("demo")


@pytest.mark.anyio
async def test_fire_returns_immediately_with_fresh_key_preserving_payload(remote):
    payload = {"n": 2}
    assert await sandbox_execution.fire_remote("demo", payload) == {
        "invocation_id": "invocation"
    }
    await sandbox_execution.fire_remote("demo", payload)
    calls = remote.fake.fire_trigger.await_args_list
    assert calls[0].args[:2] == ("trigger", {"n": 2})
    assert calls[0].args[2] != calls[1].args[2]
    assert payload == {"n": 2}
    remote.fake.list_executions.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("payload", [{}, {"n": "2"}, {"n": True}, {"n": 2, "extra": 1}])
async def test_payload_schema_validated_before_fire(remote, payload):
    with pytest.raises(SandboxRunError, match="payload"):
        await sandbox_execution.fire_remote("demo", payload)
    remote.fake.fire_trigger.assert_not_awaited()


@pytest.mark.anyio
async def test_zero_enqueued_is_not_a_successful_fire(remote):
    remote.fake.fire_trigger.return_value["enqueuedCount"] = 0
    with pytest.raises(SandboxRunError) as error:
        await sandbox_execution.fire_remote("demo", {"n": 2})
    assert error.value.invocation_id == "invocation"


@pytest.mark.anyio
async def test_fire_requires_owned_unshared_bindings(remote):
    remote.fake.list_actions.return_value = [{"actionId": "foreign"}]
    with pytest.raises(SandboxDeploymentConflict):
        await sandbox_execution.fire_remote("demo", {"n": 2})
    remote.fake.fire_trigger.assert_not_awaited()


@pytest.mark.anyio
async def test_fire_bare_function_explains_endpoint_deployment(remote):
    remote.fake.list_flows.return_value = []
    with pytest.raises(SandboxRunError, match="deploy as an endpoint"):
        await sandbox_execution.fire_remote("demo", {})


@pytest.mark.anyio
async def test_paid_guards_precede_transport_but_disable_is_free(remote):
    for action in (
        lambda: sandbox_execution.fire_remote("demo", {"n": 2}, billing="no_commerce"),
        lambda: sandbox_flows.set_enabled("demo", True, billing="no_commerce"),
    ):
        with pytest.raises(BillingError):
            await action()
    remote.fake.list_functions.assert_not_awaited()
    remote.fake.fire_trigger.assert_not_awaited()
    remote.fake.enable_flow.assert_not_awaited()
    remote.flow["enabled"] = False
    result = await sandbox_flows.set_enabled("demo", False, billing="no_commerce")
    assert result["enabled"] is False


@pytest.mark.anyio
async def test_enable_checks_ownership_and_fresh_state(remote):
    assert (await sandbox_flows.set_enabled("demo", True))["enabled"] is True
    remote.fake.list_actions.return_value = [{"actionId": "foreign"}]
    with pytest.raises(SandboxDeploymentConflict):
        await sandbox_flows.set_enabled("demo", True)
    assert remote.fake.enable_flow.await_count == 1


@pytest.mark.anyio
async def test_recent_runs_limit_and_no_read_for_nonpositive(remote):
    assert await sandbox_execution.recent_runs("demo", 0) == []
    assert await sandbox_execution.recent_runs("demo", -3) == []
    remote.fake.list_functions.assert_not_awaited()
    remote.fake.list_executions.return_value = [
        execution(),
        {**execution(), "id": "older"},
    ]
    assert (
        len(await sandbox_execution.recent_runs("demo", 1, billing="no_commerce")) == 1
    )


@pytest.mark.anyio
async def test_invocation_lookup_and_recent_runs_use_paginated_boundary(
    remote, monkeypatch
):
    sdk = SimpleNamespace(
        workflows=SimpleNamespace(
            executions=SimpleNamespace(
                list=AsyncMock(
                    side_effect=[
                        {"data": {"items": [], "pagination": {"hasNext": True}}},
                        {
                            "data": {
                                "items": [
                                    {"id": "execution", "invocationId": "invocation"}
                                ],
                                "pagination": {"hasNext": False},
                            }
                        },
                    ]
                )
            )
        )
    )

    @asynccontextmanager
    async def client(*args):
        yield sdk

    monkeypatch.setattr(clients, "_sdk", client)
    monkeypatch.setattr(clients, "list_executions", LIST_EXECUTIONS)
    assert (
        await sandbox_execution.resolve_execution(
            "demo", "invocation", deadline=sandbox_execution.time.monotonic() + 10
        )
        == "execution"
    )
    calls = sdk.workflows.executions.list.await_args_list
    assert [c.kwargs["page"] for c in calls] == [1, 2]
    assert all(c.kwargs["invocation_id"] == "invocation" for c in calls)
    sdk.workflows.executions.list.reset_mock(side_effect=True)
    sdk.workflows.executions.list.side_effect = [
        {"data": {"items": [{"id": "newest"}], "pagination": {"hasNext": True}}},
        {"data": {"items": [{"id": "older"}], "pagination": {"hasNext": False}}},
    ]
    assert [r["id"] for r in await sandbox_execution.recent_runs("demo", 2)] == [
        "newest",
        "older",
    ]
    assert (
        sdk.workflows.executions.list.await_args.kwargs["order_by_direction"] == "desc"
    )


@pytest.mark.anyio
async def test_resolve_does_not_wait_for_completion(remote):
    assert (
        await sandbox_execution.resolve_execution(
            "demo", "invocation", deadline=sandbox_execution.time.monotonic() + 10
        )
        == "execution"
    )
    remote.fake.get_execution.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("overall", "step", "expected"),
    [
        ("failed", {"status": "failed", "body": {"error": "private-token"}}, "failed"),
        ("failed", {"status": "skipped"}, "skipped"),
        ("failed", {"status": "cancelled"}, "cancelled"),
        ("cancelled", {"status": "cancelled"}, "cancelled"),
        ("success", {"body": {"eventId": "id"}}, "succeeded"),
    ],
)
async def test_webhook_outcome_preserves_function_value(
    remote, caplog, monkeypatch, overall, step, expected
):
    monkeypatch.setattr(
        logging.getLogger(sandbox_execution.__name__), "handlers", [caplog.handler]
    )
    record = execution(overall, {"result": 7})
    record["result"]["steps"].append({"uses": "webhooks.send", **step})
    remote.fake.get_execution.return_value = record
    result = await sandbox_execution.wait_result("demo", execution_id="execution")
    assert result["webhookStatus"] == expected
    assert result["result"]["steps"][0]["body"]["result"] == 7
    assert ("Sandbox webhook step" in caplog.text) == (expected != "succeeded")
    assert "private-token" not in caplog.text
    assert "private-token" not in json.dumps(result) and "private-token" not in repr(
        result
    )
    if expected == "succeeded":
        assert result["result"]["steps"][-1]["body"] == {"eventId": "id"}


@pytest.mark.anyio
async def test_success_returns_full_snapshot_with_partial_metadata(remote):
    remote.fake.get_execution.return_value = execution(
        body={
            "outcome": "partial",
            "partialReason": "budget",
            "result": {"n": 1},
            "state": {"cursor": 2},
        }
    )
    result = await sandbox_execution.wait_result("demo", invocation_id="invocation")
    assert result["id"] == "execution" and result["status"] == "success"
    assert result["result"]["steps"][0]["body"]["outcome"] == "partial"
    assert result["result"]["steps"][0]["body"]["partialReason"] == "budget"
    assert result["startedAt"] and result["finishedAt"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status", ["failed", "cancelled", "skipped", "invalid", "unknown"]
)
async def test_terminal_failure_distinct_and_safe(remote, status):
    remote.fake.get_execution.return_value = execution(
        status, {"error": "PRIVATE API KEY"}
    )
    with pytest.raises(SandboxRunError) as error:
        await sandbox_execution.wait_result("demo", execution_id="execution")
    assert status in str(error.value)
    assert "PRIVATE" not in str(error.value)
    assert error.value.execution_id == "execution"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {"outcome": "failed", "result": {}},
        {"ok": False, "result": {}},
        {"outcome": "nonsense", "result": {}},
        {},
    ],
)
async def test_success_envelope_does_not_hide_failed_or_missing_body(remote, body):
    remote.fake.get_execution.return_value = execution(body=body)
    with pytest.raises(SandboxRunError):
        await sandbox_execution.wait_result("demo", execution_id="execution")


@pytest.mark.anyio
async def test_single_deadline_shared_between_resolution_and_polling(
    remote, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(sandbox_execution.time, "monotonic", lambda: clock[0])

    async def resolve(*args, **kwargs):
        assert kwargs["deadline"] == 300.0
        clock[0] = 299.0
        return "execution"

    async def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(sandbox_execution, "resolve_execution", resolve, raising=False)
    monkeypatch.setattr(sandbox_execution.asyncio, "sleep", sleep)
    remote.fake.get_execution.return_value = execution("running")
    with pytest.raises(SandboxRunTimeout) as error:
        await sandbox_execution.wait_result("demo", invocation_id="invocation")
    assert clock[0] == 300.0
    assert (
        error.value.invocation_id == "invocation"
        and error.value.execution_id == "execution"
    )
    assert "cancel" not in str(error.value).lower()


@pytest.mark.anyio
async def test_timeout_before_execution_id_has_invocation_id(remote):
    with pytest.raises(SandboxRunTimeout) as error:
        await sandbox_execution.wait_result(
            "demo", invocation_id="invocation", timeout=0
        )
    assert (
        error.value.invocation_id == "invocation" and error.value.execution_id is None
    )
    remote.fake.list_executions.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["execution_snapshot", "recent_runs"])
@pytest.mark.parametrize(
    "overall, step_status, expected",
    [
        ("pending", "pending", "pending"),
        ("running", "running", "running"),
        ("running", None, None),
        ("success", "success", "succeeded"),
    ],
)
async def test_public_webhook_status(remote, path, overall, step_status, expected):
    record = execution(overall)
    step = {"uses": "webhooks.send", "status": step_status}
    record["progress"] = {"steps": [step]}
    record["result"]["steps"].append({"uses": "webhooks.send"})
    remote.fake.get_execution.return_value = record
    remote.fake.list_executions.return_value = [record]
    if path == "execution_snapshot":
        snapshot = await sandbox_execution.execution_snapshot("execution")
    else:
        (snapshot,) = await sandbox_execution.recent_runs("demo", 1)
    assert snapshot.get("webhookStatus") == expected
    assert snapshot["result"]["steps"][-1].get("status") == expected


@pytest.mark.anyio
async def test_public_execution_reads_allowlist_step_bodies(remote):
    record = execution(body={"result": {"n": 2}, "debug": "SECRET"})
    record["result"].update(error="SECRET")
    record["result"]["steps"].extend(
        [
            {"body": {"result": "SECRET"}},
            {
                "uses": None,
                "service": "webhooks",
                "method": "send",
                "body": {"error": "SECRET", "eventId": "evt"},
            },
        ]
    )
    remote.fake.get_execution.return_value = record
    remote.fake.list_executions.return_value = [record]
    for snapshot in [
        await sandbox_execution.execution_snapshot("execution"),
        *(await sandbox_execution.recent_runs("demo", 1)),
    ]:
        assert "SECRET" not in repr(snapshot)
        assert snapshot["result"]["steps"][0]["body"] == {"result": {"n": 2}}
        assert snapshot["webhookStatus"] == "failed"
        assert snapshot["result"]["steps"][-1]["body"] == {"eventId": "evt"}


@pytest.mark.anyio
async def test_snapshot_is_fresh_and_does_not_mutate_api_records(remote):
    before = copy.deepcopy(remote.fake.get_execution.return_value)
    snap = await sandbox_execution.execution_snapshot(
        "execution", billing="no_commerce"
    )
    snap["result"]["steps"].clear()
    assert remote.fake.get_execution.return_value == before
    remote.fake.get_execution.return_value = execution("running")
    assert (await sandbox_execution.execution_snapshot("execution"))[
        "status"
    ] == "running"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body, expected, downloaded",
    [
        (
            {"stdout": "out", "stderr": "err", "logs": "available", "runId": "compute"},
            "out\nerr",
            False,
        ),
        ({"logs": "inline"}, "inline", False),
        ({"logs": "available", "runId": "compute"}, "downloaded log", True),
        ({}, "", False),
    ],
)
async def test_logs_inline_or_downloaded_without_paid_billing(
    remote, body, expected, downloaded
):
    remote.fake.get_execution.return_value = execution(body=body)
    assert (
        await sandbox_execution.execution_logs("execution", billing="no_commerce")
        == expected
    )
    assert bool(remote.fake.run_logs.await_count) == downloaded


@pytest.mark.anyio
async def test_transport_errors_do_not_reveal_credentials(remote):
    remote.fake.fire_trigger.side_effect = RuntimeError("PRIVATE API KEY")
    with pytest.raises(SandboxRunError) as error:
        await sandbox_execution.fire_remote("demo", {"n": 2})
    assert "PRIVATE" not in str(error.value)


@pytest.mark.anyio
async def test_missing_receipt_and_shared_trigger_never_activate(remote):
    remote.store.receipt_path.unlink()
    with pytest.raises(SandboxDeploymentConflict):
        await sandbox_flows.set_enabled("demo", True)
    remote.store.write(remote.receipt)
    remote.fake.list_flows.return_value.append(
        {"id": "foreign", "name": "foreign", "triggerId": "trigger", "enabled": True}
    )
    with pytest.raises(SandboxDeploymentConflict):
        await sandbox_execution.fire_remote("demo", {"n": 2})
    remote.fake.enable_flow.assert_not_awaited()
    remote.fake.fire_trigger.assert_not_awaited()


@pytest.mark.anyio
async def test_pending_cutover_blocks_activation_but_allows_shutdown(remote):
    from dataclasses import replace

    receipt = replace(
        remote.receipt, stages=(json.dumps({"stage": "cutover", "complete": False}),)
    )
    remote.store.write(receipt)
    with pytest.raises(SandboxDeploymentConflict):
        await sandbox_flows.set_enabled("demo", True)
    with pytest.raises(SandboxDeploymentConflict):
        await sandbox_execution.fire_remote("demo", {"n": 2})
    remote.flow["enabled"] = False
    assert (await sandbox_flows.set_enabled("demo", False, billing="no_commerce"))[
        "enabled"
    ] is False


@pytest.mark.anyio
async def test_toggle_requires_fresh_confirmation_not_response(remote):
    with pytest.raises(SandboxDeploymentError, match="unconfirmed"):
        await sandbox_flows.set_enabled("demo", False)
    assert any("flow:disable" in stage for stage in remote.store.read().stages)


@pytest.mark.anyio
async def test_flow_pinned_schema_is_used_not_latest_version(remote):
    remote.document["steps"] = [
        {"uses": "functions.run", "with": {"functionId": "function", "version": 1}}
    ]
    remote.fake.get_version.return_value = {
        "version": 1,
        "status": "ready",
        "inputSchema": {"type": "object", "required": ["old"]},
    }
    with pytest.raises(SandboxRunError, match="payload"):
        await sandbox_execution.fire_remote("demo", {"n": 2})
    assert remote.fake.get_version.await_args.args == ("function", 1)
    remote.fake.fire_trigger.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("ref", ["$ref", "$dynamicRef", "$recursiveRef"])
async def test_schema_references_rejected_without_resolution(remote, ref, monkeypatch):
    resolver = Mock(side_effect=AssertionError("Do not instantiate a resolver"))
    monkeypatch.setattr("jsonschema.validators.validator_for", resolver)
    remote.document["on"]["custom"]["inputs"] = {
        ref: "https://do-not-fetch.invalid/schema"
    }
    with pytest.raises(SandboxRunError, match="payload"):
        await sandbox_execution.fire_remote("demo", {"n": 2})
    resolver.assert_not_called()
    remote.fake.fire_trigger.assert_not_awaited()


@pytest.mark.anyio
async def test_failed_function_step_in_successful_execution_raises(remote):
    remote.fake.get_execution.return_value["result"]["steps"][0]["status"] = "failed"
    with pytest.raises(SandboxRunError):
        await sandbox_execution.wait_result("demo", execution_id="execution")


@pytest.mark.anyio
async def test_explicit_null_function_result_is_valid(remote):
    remote.fake.get_execution.return_value = execution(body={"result": None})
    result = await sandbox_execution.wait_result("demo", execution_id="execution")
    assert result["result"]["steps"][0]["body"]["result"] is None


@pytest.mark.anyio
async def test_snapshot_rejects_mismatched_id_and_hides_raw_error(remote):
    remote.fake.get_execution.return_value["id"] = "different"
    remote.fake.get_execution.return_value["error"] = "PRIVATE"
    with pytest.raises(SandboxRunError):
        await sandbox_execution.execution_snapshot("execution")


@pytest.mark.anyio
@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan"), True])
async def test_invalid_timeout_never_reads_execution(remote, timeout):
    with pytest.raises(SandboxRunError):
        await sandbox_execution.wait_result(
            "demo", execution_id="execution", timeout=timeout
        )
    remote.fake.get_execution.assert_not_awaited()


@pytest.mark.anyio
async def test_read_request_duration_counts_toward_deadline(remote, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(sandbox_execution.time, "monotonic", lambda: clock[0])

    async def slow_snapshot(*args, **kwargs):
        clock[0] = 301.0
        return execution()

    monkeypatch.setattr(sandbox_execution, "execution_snapshot", slow_snapshot)
    with pytest.raises(SandboxRunTimeout) as error:
        await sandbox_execution.wait_result("demo", execution_id="execution")
    assert error.value.execution_id == "execution"


@pytest.mark.anyio
async def test_available_logs_missing_compute_id_are_not_fabricated(remote):
    remote.fake.get_execution.return_value = execution(body={"logs": "available"})
    with pytest.raises(SandboxRunError):
        await sandbox_execution.execution_logs("execution")
    remote.fake.run_logs.assert_not_awaited()


@pytest.mark.anyio
async def test_invalid_invocation_identifier_raises_safe_run_error(remote):
    remote.fake.fire_trigger.return_value["invocationId"] = "PRIVATE/bad-id"
    with pytest.raises(SandboxRunError) as error:
        await sandbox_execution.fire_remote("demo", {"n": 2})
    assert "PRIVATE" not in str(error.value)


@pytest.mark.anyio
async def test_log_request_errors_hide_credentials(remote):
    remote.fake.get_execution.side_effect = RuntimeError("PRIVATE API KEY")
    with pytest.raises(SandboxRunError) as error:
        await sandbox_execution.execution_logs("execution")
    assert "PRIVATE" not in str(error.value)


@pytest.mark.anyio
async def test_schema_property_named_format_is_not_a_format_keyword(remote):
    schema = {
        "type": "object",
        "properties": {"format": {"type": "string"}},
        "required": ["format"],
    }
    remote.document["on"]["custom"]["inputs"] = schema
    remote.fake.get_version.return_value["inputSchema"] = schema
    assert await sandbox_execution.fire_remote("demo", {"format": "text"}) == {
        "invocation_id": "invocation"
    }


@pytest.mark.anyio
async def test_overall_deadline_bounds_inflight_read(remote):
    async def never_finishes(*args, **kwargs):
        await asyncio.Event().wait()

    remote.fake.get_execution.side_effect = never_finishes
    with pytest.raises(SandboxRunTimeout) as error:
        await sandbox_execution.wait_result(
            "demo", execution_id="execution", timeout=0.01
        )
    assert error.value.execution_id == "execution"


@pytest.mark.anyio
async def test_resolve_empty_pages_until_shared_deadline_exposes_invocation(remote):
    remote.fake.list_executions.return_value = []
    with pytest.raises(SandboxRunTimeout) as error:
        await sandbox_execution.wait_result(
            "demo", invocation_id="invocation", timeout=0.01
        )
    assert (
        error.value.invocation_id == "invocation" and error.value.execution_id is None
    )
    remote.fake.get_execution.assert_not_awaited()


@pytest.mark.anyio
async def test_poll_progresses_pending_running_success_without_resetting_budget(remote):
    remote.fake.get_execution.side_effect = [
        execution("pending"),
        execution("running"),
        execution(),
    ]
    snapshot = await sandbox_execution.wait_result(
        "demo", execution_id="execution", timeout=1
    )
    assert snapshot["status"] == "success"
    assert remote.fake.get_execution.await_count == 3


@pytest.mark.anyio
async def test_envelope_error_is_not_success_even_without_outcome(remote):
    remote.fake.get_execution.return_value = execution(
        body={"error": {"code": "failed"}, "result": None}
    )
    with pytest.raises(SandboxRunError):
        await sandbox_execution.wait_result("demo", execution_id="execution")


@pytest.mark.anyio
async def test_missing_function_step_never_fires_or_enables(remote):
    remote.document.pop("steps")
    with pytest.raises(SandboxDeploymentError):
        await sandbox_execution.fire_remote("demo", {"n": 2})
    with pytest.raises(SandboxDeploymentError):
        await sandbox_flows.set_enabled("demo", True)
    remote.fake.fire_trigger.assert_not_awaited()
    remote.fake.enable_flow.assert_not_awaited()


def test_sandbox_extra_declares_plain_validator_without_optional_formats():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert "jsonschema>=4,<5" in project["project"]["optional-dependencies"]["sandbox"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "field,format_name,value",
    [
        ("at", "date-time", "2026-11-24T10:00:00Z"),
        ("at", "date-time", "not-a-date"),
        ("id", "uuid", "12345678-1234-1234-1234-123456789abc"),
        ("id", "uuid", "not-a-uuid"),
    ],
    ids=["datetime-valid", "datetime-invalid", "uuid-valid", "uuid-invalid"],
)
async def test_format_validation_is_deferred_to_server(
    remote, field, format_name, value
):
    schema = {
        "type": "object",
        "required": [field],
        "properties": {field: {"type": "string", "format": format_name}},
    }
    remote.document["on"]["custom"]["inputs"] = schema
    remote.fake.get_version.return_value["inputSchema"] = schema
    assert (await sandbox_execution.fire_remote("demo", {field: value}))[
        "invocation_id"
    ] == "invocation"


@pytest.mark.anyio
@pytest.mark.parametrize("keyword", ["default", "examples", "enum", "const"])
async def test_schema_instance_annotations_are_not_walked_as_schemas(remote, keyword):
    instance = {"format": "application-specific", "$ref": "literal instance string"}
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        keyword: [instance] if keyword in ("examples", "enum") else instance,
    }
    payload = instance if keyword in ("enum", "const") else {"n": 2}
    remote.document["on"]["custom"]["inputs"] = schema
    remote.fake.get_version.return_value["inputSchema"] = schema
    assert (await sandbox_execution.fire_remote("demo", payload))[
        "invocation_id"
    ] == "invocation"


@pytest.mark.anyio
async def test_scheduled_flow_rejects_fire_without_mutation(remote):
    remote.document["on"] = {"schedule": {"cron": "* * * * *", "timezone": "UTC"}}
    with pytest.raises(SandboxRunError, match="schedule"):
        await sandbox_execution.fire_remote("demo", {})
    remote.fake.fire_trigger.assert_not_awaited()
    assert not (await sandbox_flows.describe_remote("demo")).get("endpoint")


@pytest.mark.anyio
async def test_legacy_custom_trigger_without_endpoint_schema_cannot_fire(remote):
    remote.document["on"] = {"custom": {}}
    remote.fake.get_version.return_value.pop("inputSchema")
    with pytest.raises(SandboxRunError, match="deploy as an endpoint"):
        await sandbox_execution.fire_remote("demo", {})
    remote.fake.fire_trigger.assert_not_awaited()
    assert not (await sandbox_flows.describe_remote("demo")).get("endpoint")
