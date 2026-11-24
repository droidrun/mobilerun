import asyncio
import importlib
import json
import time
from unittest.mock import AsyncMock

import pytest
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    Field,
    ValidationError,
    field_serializer,
)

import mobilerun.sandbox as sandbox
from mobilerun.models.billing import Billing
from mobilerun.sandbox.errors import (
    SandboxDeclarationError,
    SandboxDeploymentError,
    SandboxRunError,
    SandboxRunTimeout,
)


class Input(BaseModel):
    n: int = Field(1, serialization_alias="wireN")


class Output(BaseModel):
    n: int


@pytest.fixture
def backend(monkeypatch):
    operations = importlib.import_module("mobilerun.sandbox.deployment")
    snapshot = {
        "id": "run",
        "status": "success",
        "startedAt": "2026-01-01T00:00:00Z",
        "finishedAt": "2026-01-01T00:00:02Z",
        "invocationId": "inv",
        "result": {"steps": [{"uses": "functions.run", "body": {"result": {"n": 2}}}]},
    }
    defaults = {
        "describe_remote": {
            "function_id": "fn",
            "version": 2,
            "flow_id": "flow",
            "endpoint": {
                "url": "https://example.test/v1/triggers/t/fire",
                "trigger_id": "t",
            },
        },
        "fire_remote": {"invocation_id": "inv"},
        "recent_runs": [snapshot],
        "set_enabled": {"id": "flow"},
        "execution_snapshot": snapshot,
        "resolve_execution": "run",
        "wait_result": snapshot,
        "execution_logs": "[INFO] test",
    }
    fakes = {name: AsyncMock(return_value=value) for name, value in defaults.items()}
    for name, fake in fakes.items():
        monkeypatch.setattr(operations, name, fake)
    return fakes


def test_lazy_lookup_refreshes_after_redeploy(backend):
    handle = sandbox.cloud["x"]
    assert all(not fake.called for fake in backend.values())
    assert (handle.function_id, handle.version, handle.flow_id) == ("fn", 2, "flow")
    assert handle.endpoint == sandbox.Endpoint(
        "https://example.test/v1/triggers/t/fire", "t"
    )
    backend["describe_remote"].return_value = {
        "function_id": "fn",
        "version": 3,
        "flow_id": "new-flow",
        "endpoint": {
            "url": "https://example.test/v1/triggers/new/fire",
            "trigger_id": "new",
        },
    }
    assert (handle.version, handle.flow_id, handle.endpoint.trigger_id) == (
        3,
        "new-flow",
        "new",
    )


def test_typed_and_untyped_run(backend):
    handle = sandbox.Remote("x", Input, Output)
    with pytest.raises(ValidationError):
        handle(n="wrong")
    backend["fire_remote"].assert_not_called()
    run = handle.fire(n="2")
    assert run.invocation_id == "inv"
    assert run.id == "run"
    assert run.status == "success"
    assert run.result() == Output(n=2)
    assert "[INFO] test" in run.logs()
    assert sandbox.cloud["x"](n=2) == {"n": 2}


@pytest.mark.parametrize("with_validation_alias", [False, True])
def test_fire_does_not_wait_and_payload_matches_worker(backend, with_validation_alias):
    class Aliased(BaseModel):
        n: int = Field(validation_alias="incoming", serialization_alias="outgoing")

    model = Aliased if with_validation_alias else Input
    value = Aliased(incoming=4) if with_validation_alias else Input(n=4)
    run = sandbox.Remote("x", model).fire(value)
    name, payload = backend["fire_remote"].await_args.args
    assert name == "x" and model.model_validate(payload).n == 4
    assert set(payload) == set(model.model_json_schema(mode="validation")["properties"])
    assert run.invocation_id == "inv"
    backend["resolve_execution"].assert_not_called()
    backend["wait_result"].assert_not_called()


@pytest.mark.parametrize("choice", [False, True])
def test_typed_fire_preserves_nested_validation_aliases(backend, choice):
    path = AliasPath("payload", 0, "n")
    alias = AliasChoices(path, "fallback") if choice else path

    class Payload(BaseModel):
        n: int = Field(1, validation_alias=alias)
        m: int = Field(2, validation_alias=AliasPath("payload", 0, "m"))

    expected = {"payload": [{"n": 4, "m": 5}]}
    original = Payload.model_validate(
        {"fallback": 4, "payload": [{"m": 5}]} if choice else expected
    )
    sandbox.Remote("x", Payload).fire(original)
    _, payload = backend["fire_remote"].await_args.args
    assert payload == expected
    assert Payload.model_validate_json(json.dumps(payload)) == original


def test_typed_fire_rejects_lossy_serialization_before_transport(backend):
    class Lossy(BaseModel):
        n: int

        @field_serializer("n")
        def serialize_n(self, value: int) -> int:
            return value + 1

    with pytest.raises(SandboxDeclarationError, match="round-trip"):
        sandbox.Remote("x", Lossy).fire(Lossy(n=4))
    backend["fire_remote"].assert_not_called()


def test_payload_billing_is_distinct_from_control_and_override_sticks(
    backend, monkeypatch
):
    monkeypatch.setenv("MOBILERUN_BILLING", "no_commerce")
    handle = sandbox.cloud["x"]
    monkeypatch.setenv("MOBILERUN_BILLING", "paid")
    _ = handle.function_id
    assert (
        backend["describe_remote"].await_args.kwargs["billing"] == Billing.NO_COMMERCE
    )
    run = handle.fire(input={"billing": "customer-value"}, billing="paid")
    backend["fire_remote"].assert_awaited_once_with(
        "x", {"billing": "customer-value"}, billing=Billing.PAID
    )
    assert run.result() == {"n": 2}
    assert backend["wait_result"].await_args.kwargs["billing"] == Billing.PAID
    with pytest.raises(TypeError):
        handle.fire(input={}, n=2)
    backend["fire_remote"].assert_awaited_once()


def test_recent_runs_toggle_and_snapshot_copy(backend):
    handle = sandbox.cloud["x"]
    assert handle.runs(0) == []
    backend["recent_runs"].assert_not_called()
    (run,) = handle.runs(limit=1, billing="no_commerce")
    assert run.id == "run"
    assert run.invocation_id == "inv"
    assert run.started_at == "2026-01-01T00:00:00Z"
    assert run.ended_at == "2026-01-01T00:00:02Z"
    assert run.duration == 2
    run.snapshot["result"]["steps"].clear()
    assert run.snapshot["result"]["steps"]
    _ = run.status
    assert (
        backend["execution_snapshot"].await_args.kwargs["billing"]
        == Billing.NO_COMMERCE
    )
    handle.disable(billing="no_commerce")
    handle.enable(billing="paid")
    assert [call.args for call in backend["set_enabled"].await_args_list] == [
        ("x", False),
        ("x", True),
    ]


@pytest.mark.parametrize("property_name", ["function_id", "flow_id", "endpoint"])
def test_missing_deployment_explains_cli(backend, property_name):
    if property_name == "function_id":
        backend["describe_remote"].side_effect = SandboxDeploymentError(
            "No matching function", missing_deployment=True
        )
    else:
        backend["describe_remote"].return_value = {
            "function_id": "fn",
            "version": 2,
            "flow_id": None,
        }
    with pytest.raises(SandboxDeploymentError, match="mobilerun sandbox deploy FILE"):
        getattr(sandbox.cloud["x"], property_name)


def test_no_enqueued_execution_surfaces_boundary_error(backend):
    backend["fire_remote"].side_effect = SandboxRunError(
        "did not enqueue", invocation_id="inv"
    )
    with pytest.raises(SandboxRunError, match="did not enqueue") as error:
        sandbox.cloud["x"].fire()
    assert error.value.invocation_id == "inv"


@pytest.mark.parametrize("known", [False, True])
def test_result_timeout_is_passed_to_shared_waiter(backend, known):
    backend["wait_result"].side_effect = SandboxRunTimeout(
        "deadline exceeded", invocation_id="inv"
    )
    run = sandbox.Run(
        sandbox.cloud["x"], invocation_id="inv", execution_id="run" if known else None
    )
    with pytest.raises(SandboxRunTimeout):
        run.result(timeout=1)
    assert backend["wait_result"].await_args.kwargs["timeout"] == 1
    assert backend["wait_result"].await_args.kwargs["execution_id"] == (
        "run" if known else None
    )


def test_lazy_id_resolution_has_a_deadline(backend):
    run = sandbox.cloud["x"].fire()
    assert run.id == run.id == "run"
    backend["resolve_execution"].assert_awaited_once()
    assert backend["resolve_execution"].await_args.kwargs["deadline"] > time.monotonic()


def test_run_uses_configured_timeout(backend, monkeypatch):
    monkeypatch.setenv("MOBILERUN_SANDBOX_RUN_TIMEOUT", "17")
    run = sandbox.cloud["x"].fire()
    start = time.monotonic()
    assert run.id == "run"
    assert backend["resolve_execution"].await_args.kwargs["deadline"] == pytest.approx(
        start + 17, abs=1
    )
    run.result()
    assert backend["wait_result"].await_args.kwargs["timeout"] == 17


@pytest.mark.parametrize(
    "value,model,expected",
    [
        (None, Output | None, None),
        ({"n": 2}, Output | None, Output(n=2)),
        ([{"n": 2}], list[Output], [Output(n=2)]),
    ],
)
def test_nullable_and_generic_result(backend, value, model, expected):
    backend["wait_result"].return_value["result"]["steps"][0]["body"]["result"] = value
    assert sandbox.Remote("x", output_type=model).fire().result() == expected


def test_run_result_propagates_failed_body_and_preserves_partial(monkeypatch):
    operations = importlib.import_module("mobilerun.sandbox.deployment")
    handle = sandbox.Remote("x", output_type=Output)
    body = {"outcome": "failed", "result": {"n": 2}}
    snapshot = {
        "id": "run",
        "status": "success",
        "result": {"steps": [{"uses": "functions.run", "body": body}]},
    }
    monkeypatch.setattr(
        operations, "execution_snapshot", AsyncMock(return_value=snapshot)
    )
    run = sandbox.Run(handle, execution_id="run")
    with pytest.raises(SandboxRunError, match="body failed"):
        run.result()
    body.update(outcome="partial", notes=["limited"])
    assert run.result() == Output(n=2)
    assert run.snapshot["result"]["steps"][0]["body"]["outcome"] == "partial"
    assert run.snapshot["result"]["steps"][0]["body"]["notes"] == ["limited"]


@pytest.mark.parametrize("has_tail", [True, False])
def test_logs_uses_inline_tail_or_boundary_download(monkeypatch, has_tail):
    operations = importlib.import_module("mobilerun.sandbox.deployment")
    body = {"logs": "available", "runId": "compute"}
    if has_tail:
        body["stderr"] = "inline tail"
    snapshot = {
        "id": "run",
        "result": {"steps": [{"uses": "functions.run", "body": body}]},
    }
    monkeypatch.setattr(
        operations, "execution_snapshot", AsyncMock(return_value=snapshot)
    )
    download = AsyncMock(return_value="downloaded logs")
    monkeypatch.setattr(operations.clients, "run_logs", download)
    expected = "inline tail" if has_tail else "downloaded logs"
    assert sandbox.Run(sandbox.cloud["x"], execution_id="run").logs() == expected
    if has_tail:
        download.assert_not_called()
    else:
        assert download.await_args.args == ("compute",)


def test_mutations_redelegate_by_name_without_retrying_cached_ids(backend):
    handle = sandbox.Remote("x", billing="paid")
    assert handle.flow_id == "flow"
    handle.disable()
    backend["set_enabled"].assert_awaited_once_with("x", False, billing=Billing.PAID)
    backend["fire_remote"].side_effect = SandboxRunError("request failed")
    with pytest.raises(SandboxRunError, match="request failed"):
        handle.fire()
    backend["fire_remote"].assert_awaited_once_with("x", {}, billing=Billing.PAID)
    backend["describe_remote"].assert_awaited_once()


def test_active_event_loop_does_not_create_coroutine(backend, recwarn):
    async def attempt():
        with pytest.raises(RuntimeError, match="event loop"):
            sandbox.cloud["x"].fire(n=2)

    asyncio.run(attempt())
    backend["fire_remote"].assert_not_called()
    assert not recwarn.list
