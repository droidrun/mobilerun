import asyncio
import importlib
import json
import time
from functools import partial
from pathlib import Path
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
from mobilerun.clients.cloud.billing import with_billing_retry
from mobilerun.models.billing import Billing
from mobilerun.sandbox.errors import (
    SandboxDeclarationError,
    SandboxDeploymentError,
    SandboxRunError,
    SandboxRunTimeout,
)
from tests.support.sandbox_fixtures import Input, Output
from tests.support.sandbox_fixtures import backend as backend


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


@pytest.mark.parametrize("operation", ["fire", "enable", "call"])
def test_remote_mutations_retry_with_paid(backend, monkeypatch, operation):
    from mobilerun.sandbox import app
    from mobilerun.sandbox.billing import require_paid

    prompt = AsyncMock(return_value=True)
    monkeypatch.setattr(
        app,
        "with_billing_retry",
        partial(with_billing_retry, prompt=prompt, is_interactive=lambda: True),
    )
    handle = sandbox.cloud["x"]
    name = "set_enabled" if operation == "enable" else "fire_remote"
    fake = backend[name]

    async def guarded(*args, billing):
        require_paid(billing=billing)
        return fake.return_value

    fake.side_effect = guarded
    if operation == "enable":
        handle.enable(billing="included", yes=True)
    else:
        if operation == "fire":
            assert handle.fire(billing="included", n=2).result() == {"n": 2}
        else:
            assert handle(billing="included", n=2) == {"n": 2}
    assert [call.kwargs["billing"] for call in fake.await_args_list] == [
        Billing.INCLUDED,
        Billing.PAID,
    ]
    if operation != "enable":
        prompt.assert_awaited_once()
        assert backend["wait_result"].await_args.kwargs["billing"] is Billing.PAID
    else:
        prompt.assert_not_awaited()


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("method", ["fire", "__call__"])
def test_remote_yes_keyword_is_payload(backend, typed, method):
    class Vote(BaseModel):
        yes: bool

    handle = sandbox.Remote("x", Vote) if typed else sandbox.cloud["x"]
    getattr(handle, method)(yes=True, billing="paid")
    backend["fire_remote"].assert_awaited_once_with(
        "x", {"yes": True}, billing=Billing.PAID
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
    execution = importlib.import_module("mobilerun.sandbox.execution")
    handle = sandbox.Remote("x", output_type=Output)
    body = {"outcome": "failed", "result": {"n": 2}}
    snapshot = {
        "id": "run",
        "status": "success",
        "result": {"steps": [{"uses": "functions.run", "body": body}]},
    }
    monkeypatch.setattr(
        execution, "execution_snapshot", AsyncMock(return_value=snapshot)
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
        operations.clients, "get_execution", AsyncMock(return_value=snapshot)
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


def test_explicit_deploy_is_exported_even_after_loading_submodule(monkeypatch):
    import mobilerun.sandbox.app as app
    import mobilerun.sandbox.deployment as operations

    receipt = object()
    fake = AsyncMock(return_value=receipt)
    artifact = object()
    monkeypatch.setattr(app, "package", lambda _: artifact)
    monkeypatch.setattr(operations, "deploy_file", fake)
    monkeypatch.setenv("MOBILERUN_BILLING", "paid")

    assert sandbox.deploy("main.py", device_free=True) is receipt
    fake.assert_awaited_once_with(
        Path("main.py"),
        device=None,
        device_free=True,
        billing=Billing.PAID,
        artifact=artifact,
    )
    assert sandbox.deploy("main.py", billing="paid", yes=True) is receipt
    assert fake.await_args.kwargs["billing"] == Billing.PAID


@pytest.mark.parametrize(
    ("mode", "tty", "answer", "yes", "attempts"),
    [
        (Billing.PAID, True, None, False, [Billing.PAID]),
        (Billing.INCLUDED, True, "y", False, [Billing.INCLUDED, Billing.PAID]),
        (Billing.INCLUDED, True, "n", False, [Billing.INCLUDED]),
        (Billing.NO_COMMERCE, False, None, False, [Billing.NO_COMMERCE]),
        (Billing.NO_COMMERCE, False, None, True, [Billing.NO_COMMERCE, Billing.PAID]),
    ],
)
def test_sdk_deploy_escalation(monkeypatch, mode, tty, answer, yes, attempts):
    from mobilerun.errors import BillingRequired
    from mobilerun.sandbox import app, deployment
    from mobilerun.sandbox.billing import require_paid

    seen = []
    prompts = []
    packaged = []
    inspected = object()

    def package(path):
        assert attempts[-1] is Billing.PAID, "rejected deploy must not inspect source"
        assert prompts == ([] if answer is None else [answer])
        packaged.append(path)
        return inspected

    async def deploy(path, *, device, device_free, billing, artifact):
        assert artifact is inspected
        seen.append(billing)
        require_paid(billing=billing)
        return "created"

    async def prompt(_error):
        assert answer is not None
        prompts.append(answer)
        return answer == "y"

    monkeypatch.setattr(
        app,
        "with_billing_retry",
        partial(with_billing_retry, prompt=prompt, is_interactive=lambda: tty),
    )
    monkeypatch.setattr(deployment, "deploy_file", deploy)
    monkeypatch.setattr(app, "package", package)
    if attempts[-1] is Billing.PAID:
        assert sandbox.deploy("main.py", billing=mode, yes=yes) == "created"
    else:
        with pytest.raises(BillingRequired, match="yes=True") as caught:
            sandbox.deploy("main.py", billing=mode, yes=yes)
        assert (caught.value.given, caught.value.needed) == (mode, Billing.PAID)
    accepted = attempts[-1] is Billing.PAID
    assert seen == ([Billing.PAID] if accepted else [])
    assert packaged == ([Path("main.py")] if accepted else [])
    assert prompts == ([] if answer is None else [answer])


@pytest.mark.parametrize(
    "name", ["App", "Task", "SandboxNotImplemented", "DeployedScript", "RunResult"]
)
def test_obsolete_public_names_are_gone(name):
    assert not hasattr(sandbox, name)


def test_obsolete_client_is_gone():
    assert importlib.util.find_spec("mobilerun.sandbox.client") is None


def test_obsolete_sandbox_error_class_is_gone():
    assert not hasattr(
        importlib.import_module("mobilerun.sandbox.errors"), "SandboxNotImplemented"
    )


def test_python_deploy_rejects_blank_device_before_packaging(monkeypatch):
    from mobilerun.sandbox import app
    from mobilerun.sandbox.errors import SandboxDeclarationError

    def tripwire(path):
        pytest.fail("invalid device reached packaging")

    monkeypatch.setattr(app, "package", tripwire)
    with pytest.raises(SandboxDeclarationError, match="Device"):
        sandbox.deploy("main.py", device=" \t", yes=True)


def test_deploy_file_previews_supplied_artifact_without_rereading(
    tmp_path, monkeypatch
):
    from mobilerun.errors import BillingRequired
    from mobilerun.sandbox import deployment
    from mobilerun.sandbox.models import Artifact, CloudDeclaration

    path = tmp_path / "changed.py"
    path.write_text("not the confirmed source")
    artifact = Artifact(CloudDeclaration("confirmed"), "", "digest", {}, None)
    monkeypatch.setattr(
        deployment, "package", lambda _: pytest.fail("unexpected repackaging")
    )

    with pytest.raises(BillingRequired):
        asyncio.run(
            deployment.deploy_file(path, artifact=artifact, billing="no_commerce")
        )
