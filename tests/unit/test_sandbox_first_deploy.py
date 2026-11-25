"""First deploy and immutable version orchestration with fake cloud APIs."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.sandbox import deployment as deploy
from mobilerun.sandbox.errors import (
    FlowStagingError,
    SandboxConfirmationRequired,
    SandboxDeclarationError,
    SandboxDeploymentConflict,
    SandboxDeploymentError,
)
from mobilerun.sandbox.models import Artifact, CloudDeclaration, OwnershipReceipt
from mobilerun.sandbox.settings import SandboxSettings
from tests.support.sandbox_fakes import InitialSandboxCloud, ProcessInterrupted


@pytest.fixture
def env(tmp_path, monkeypatch):
    artifact = Artifact(
        CloudDeclaration("demo"),
        "new source",
        "digest",
        {"type": "object"},
        None,
    )
    monkeypatch.setattr(deploy, "package", lambda path: artifact, raising=False)
    scope = deploy.Scope("https://example.test/v1", "a" * 64)
    monkeypatch.setattr(deploy, "current_scope", lambda: scope)
    settings = SandboxSettings(
        MOBILERUN_SANDBOX_DIRECTORY=str(tmp_path), MOBILERUN_SANDBOX_POLL_INTERVAL=0.001
    )
    monkeypatch.setattr(deploy, "get_sandbox_settings", lambda: settings)
    fake = SimpleNamespace(
        list_functions=AsyncMock(return_value=[]),
        list_flows=AsyncMock(return_value=[]),
        list_versions=AsyncMock(return_value=[]),
        get_source=AsyncMock(return_value="new source"),
        create_function=AsyncMock(
            return_value={"id": "f", "latestVersion": {"version": 1}}
        ),
        create_version=AsyncMock(return_value={"version": 2}),
        get_version=AsyncMock(return_value={"version": 1, "status": "ready"}),
        validate_document=AsyncMock(return_value={"valid": True}),
        create_document=AsyncMock(
            return_value={
                "flowId": "flow",
                "triggerId": "trigger",
                "actionIds": ["action"],
            }
        ),
        enable_flow=AsyncMock(return_value={"id": "flow", "enabled": True}),
    )

    def enable(flow_id, **kwargs):
        flow = fake.enable_flow.return_value
        fake.list_flows.return_value = [flow]
        return flow

    fake.enable_flow.side_effect = enable
    monkeypatch.setattr(deploy, "clients", fake, raising=False)
    return SimpleNamespace(
        artifact=artifact,
        scope=scope,
        settings=settings,
        fake=fake,
        path=Path("example.py"),
        store=deploy.ReceiptStore(scope, "demo", settings),
    )


def owned(e, *, flow=False):
    r = OwnershipReceipt(
        e.scope.origin,
        e.scope.account_id,
        "demo",
        "f",
        "flow" if flow else None,
        "trigger" if flow else None,
        ("action",) if flow else (),
        None,
        (),
    )
    e.store.write(r)
    e.fake.list_functions.return_value = [
        {"id": "f", "name": "demo", "latestVersion": {"version": 1}}
    ]
    e.fake.list_versions.return_value = [{"version": 1, "status": "ready"}]
    return r


@pytest.mark.anyio
async def test_consent_before_any_cloud_call(env):
    for kwargs in ({}, {"confirm": lambda preview: False}):
        with pytest.raises(SandboxConfirmationRequired):
            await deploy.deploy_file(env.path, **kwargs)
    env.fake.list_functions.assert_not_awaited()
    assert not env.store.directory.exists()


@pytest.mark.anyio
async def test_device_selection_and_invalid_schema_precede_cloud(env, monkeypatch):
    schedule = CloudDeclaration("demo", schedule="* * * * *")
    monkeypatch.setattr(
        deploy, "package", lambda path: replace(env.artifact, declaration=schedule)
    )
    with pytest.raises(SandboxDeclarationError):
        await deploy.deploy_file(env.path, yes=True)
    monkeypatch.setattr(
        deploy,
        "package",
        lambda path: replace(env.artifact, input_schema={"$ref": "#/$defs/X"}),
    )
    with pytest.raises(SandboxDeclarationError):
        await deploy.deploy_file(env.path, yes=True)
    env.fake.list_functions.assert_not_awaited()


@pytest.mark.anyio
async def test_duplicate_and_foreign_function_rejected(env):
    env.fake.list_functions.return_value = [
        {"id": "f", "name": "demo"},
        {"id": "g", "name": "demo"},
    ]
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(env.path, yes=True)
    env.fake.list_functions.return_value = [{"id": "f", "name": "demo"}]
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(env.path, yes=True)
    env.fake.create_function.assert_not_awaited()
    env.fake.create_document.assert_not_awaited()


@pytest.mark.anyio
async def test_new_endpoint_stages_persists_and_enables(env):
    result = await deploy.deploy_file(env.path, yes=True)
    assert (result.status, result.function_id, result.version, result.flow_id) == (
        "created",
        "f",
        1,
        "flow",
    )
    assert result.endpoint.trigger_id == "trigger"
    assert env.store.read().action_ids == ("action",)
    assert '"on":' in env.store.yaml_path.read_text()
    env.fake.enable_flow.assert_awaited_once()


@pytest.mark.anyio
async def test_staging_failure_records_partial_ids_and_no_yaml(env):
    env.fake.create_document.side_effect = FlowStagingError("trigger", ["action"], None)
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.deploy_file(env.path, yes=True)
    assert error.value.receipt.trigger_id == "trigger"
    assert env.store.read().action_ids == ("action",)
    assert not env.store.yaml_path.exists()
    env.fake.enable_flow.assert_not_awaited()


@pytest.mark.anyio
async def test_identical_ready_reuses_version(env):
    owned(env)
    result = await deploy.deploy_file(env.path, yes=True)
    assert result.status == "unchanged" and result.version == 1
    env.fake.create_version.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["rejected", "pending", "pending_inspection"])
async def test_same_source_not_ready_never_activates(env, status, monkeypatch):
    owned(env)
    env.fake.get_version.return_value = {"version": 1, "status": status}
    times = iter([0.0, 121.0])
    monkeypatch.setattr(deploy.time, "monotonic", lambda: next(times, 121.0))
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(env.path, yes=True)
    env.fake.enable_flow.assert_not_awaited()


@pytest.mark.anyio
async def test_changed_source_creates_and_polls_exact_version(env):
    owned(env)
    env.fake.get_source.return_value = "old source"
    env.fake.get_version.side_effect = [
        {"version": 1, "status": "ready"},
        {"version": 2, "status": "ready"},
    ]
    result = await deploy.deploy_file(env.path, yes=True)
    assert (result.status, result.version) == ("pushed", 2)
    assert env.fake.get_version.await_args_list[-1].args == ("f", 2)


@pytest.mark.anyio
async def test_enable_failure_exposes_persisted_receipt(env):
    env.fake.enable_flow.side_effect = RuntimeError("sensitive upstream detail")
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.deploy_file(env.path, yes=True)
    assert error.value.receipt.flow_id == "flow"
    assert "sensitive" not in str(error.value)
    assert env.store.read().flow_id == "flow"


@pytest.mark.anyio
async def test_pending_inspection_eventually_ready(env):
    env.fake.get_version.side_effect = [
        {"version": 1, "status": "pending_inspection"},
        {"version": 1, "status": "ready"},
    ]
    result = await deploy.deploy_file(env.path, yes=True)
    assert result.version == 1


@pytest.mark.anyio
async def test_malformed_staging_response_keeps_known_ids(env):
    env.fake.create_document.return_value = {"flowId": "flow", "triggerId": "trigger"}
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(env.path, yes=True)
    assert env.store.read().flow_id == "flow"
    assert env.store.read().trigger_id == "trigger"
    assert not env.store.yaml_path.exists()
    env.fake.enable_flow.assert_not_awaited()


@pytest.mark.anyio
async def test_explicit_device_free_schedule_is_staged_and_enabled(env, monkeypatch):
    declaration = CloudDeclaration("demo", schedule="* * * * *")
    monkeypatch.setattr(
        deploy, "package", lambda path: replace(env.artifact, declaration=declaration)
    )
    seen = []
    result = await deploy.deploy_file(
        env.path, device_free=True, confirm=lambda preview: seen.append(preview) or True
    )
    doc = env.fake.create_document.await_args.args[0]
    assert "devices" not in doc
    assert seen[0].device is None
    assert result.endpoint is None
    env.fake.enable_flow.assert_awaited_once()


@pytest.mark.anyio
async def test_flow_validation_failure_does_not_stage(env):
    env.fake.validate_document.return_value = {"valid": False}
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.deploy_file(env.path, yes=True)
    assert error.value.receipt.function_id == "f"
    env.fake.create_document.assert_not_awaited()
    assert not env.store.yaml_path.exists()


@pytest.mark.anyio
async def test_unowned_flow_blocks_before_creating_function(env):
    env.fake.list_flows.return_value = [{"id": "other", "name": "demo · flow"}]
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(env.path, yes=True)
    env.fake.create_function.assert_not_awaited()
    assert not env.store.yaml_path.exists()


@pytest.mark.anyio
async def test_unconfirmed_enable_is_not_reported_as_success(env):
    env.fake.enable_flow.return_value = {"id": "flow", "enabled": False}
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.deploy_file(env.path, yes=True)
    assert error.value.receipt.flow_id == "flow"


@pytest.fixture
def initial_cloud(env, monkeypatch):
    cloud = InitialSandboxCloud(
        deploy.build_document(env.artifact, "f", 1, device=None)
    )
    monkeypatch.setattr(deploy, "clients", cloud)
    env.cloud = cloud
    return env


@pytest.mark.anyio
@pytest.mark.parametrize("stage", ["function", "trigger", "action", "flow"])
@pytest.mark.parametrize("failure", ["lost", "interrupt"])
async def test_initial_creation_recovers_original_keys_not_visible_names(
    initial_cloud, stage, failure
):
    e = initial_cloud
    setattr(e.cloud, "fail" if failure == "lost" else "interrupt", stage)
    with pytest.raises(
        SandboxDeploymentError if failure == "lost" else ProcessInterrupted
    ):
        await deploy.deploy_file(e.path, yes=True)
    assert e.store.read().stages
    first_keys = dict(e.cloud.create_calls)
    result = await deploy.deploy_file(e.path, yes=True)
    assert result.function_id == "f" and result.flow_id == "flow"
    calls = [key for resource, key in e.cloud.create_calls if resource == stage]
    assert len(calls) == 2 and calls[0] == calls[1] == first_keys[stage]
    assert len(e.cloud.created) == 4
    assert e.cloud.flow["enabled"] is True


@pytest.mark.anyio
async def test_initial_unknown_action_blocks_function_cleanup(initial_cloud):
    e = initial_cloud
    e.cloud.fail = "action"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(e.path, yes=True)
    receipt = e.store.read()
    assert receipt.trigger_id == "trigger" and not receipt.action_ids
    with pytest.raises(SandboxDeploymentError, match="creation.*uncertain"):
        await deploy.cleanup_owned(receipt, delete_function=True)
    assert e.cloud.function_exists
    assert "delete_function" not in e.cloud.calls


@pytest.mark.anyio
async def test_pending_initial_intent_cannot_be_changed(initial_cloud, monkeypatch):
    e = initial_cloud
    e.cloud.fail = "flow"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(e.path, yes=True)
    before = list(e.cloud.create_calls)
    monkeypatch.setattr(
        deploy,
        "package",
        lambda p: replace(e.artifact, source="different", digest="different"),
    )
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(e.path, yes=True)
    assert e.cloud.create_calls == before


@pytest.mark.anyio
async def test_ready_response_after_inspection_deadline_is_rejected(env, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(deploy.time, "monotonic", lambda: clock[0])

    async def late(*args, **kwargs):
        clock[0] = 121.0
        return {"version": 1, "status": "ready"}

    env.fake.get_version.side_effect = late
    with pytest.raises(SandboxDeploymentError, match="timed out"):
        await deploy._wait_ready("f", 1, deploy.Billing.PAID, owned(env))


@pytest.mark.anyio
async def test_inspection_request_is_bounded_by_remaining_deadline(env, monkeypatch):
    import asyncio

    original = asyncio.wait_for
    budgets = []

    async def bounded(awaitable, timeout):
        budgets.append(timeout)
        return await original(awaitable, min(timeout, 0.01))

    async def slow(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(deploy.asyncio, "wait_for", bounded)
    env.fake.get_version.side_effect = slow
    # The outer test guard fails fast if production forgets to bound the request.
    with pytest.raises(SandboxDeploymentError, match="timed out"):
        await original(deploy._wait_ready("f", 1, deploy.Billing.PAID, owned(env)), 0.1)
    assert len(budgets) == 1 and 0 < budgets[0] <= 120


@pytest.mark.anyio
async def test_endpoint_default_instance_ref_is_not_a_schema_reference(
    env, monkeypatch
):
    monkeypatch.setattr(
        deploy,
        "package",
        lambda p: replace(
            env.artifact,
            input_schema={
                "type": "object",
                "default": {"$ref": "literal", "format": "app-data"},
            },
        ),
    )
    result = await deploy.deploy_file(env.path, yes=True)
    assert result.flow_id == "flow"


@pytest.mark.anyio
async def test_interrupted_receipt_export_cannot_activate_until_replayed(
    initial_cloud, monkeypatch
):
    e = initial_cloud
    original = deploy.ReceiptStore.write_yaml

    def stop(*args, **kwargs):
        raise ProcessInterrupted()

    monkeypatch.setattr(deploy.ReceiptStore, "write_yaml", stop)
    with pytest.raises(ProcessInterrupted):
        await deploy.deploy_file(e.path, yes=True)
    assert e.store.read().flow_id == "flow" and e.store.read().action_ids == ("action",)
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.set_enabled("demo", True)
    assert "enable" not in e.cloud.calls
    monkeypatch.setattr(deploy.ReceiptStore, "write_yaml", original)
    await deploy.deploy_file(e.path, yes=True)
    assert e.cloud.flow["enabled"] is True


@pytest.mark.anyio
@pytest.mark.parametrize("resource", ["function", "flow"])
async def test_pending_create_does_not_adopt_conflicting_visible_name(
    initial_cloud, monkeypatch, resource
):
    e = initial_cloud
    e.cloud.fail = resource
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(e.path, yes=True)
    if resource == "function":
        monkeypatch.setattr(
            e.cloud,
            "list_functions",
            AsyncMock(return_value=[{"id": "foreign", "name": "demo"}]),
        )
    else:
        monkeypatch.setattr(
            e.cloud,
            "list_flows",
            AsyncMock(return_value=[{**e.cloud.flow, "id": "foreign"}]),
        )
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(e.path, yes=True)
    keys = [key for stage, key in e.cloud.create_calls if stage == resource]
    assert len(keys) == 2 and keys[0] == keys[1]
    assert "enable" not in e.cloud.calls
    assert e.store.read().function_id == "f"
    if resource == "flow":
        assert e.store.read().flow_id == "flow"


@pytest.mark.anyio
async def test_cleanup_initial_stage_with_all_creation_ids_known_is_allowed(
    initial_cloud, monkeypatch
):
    e = initial_cloud

    def stop(*args, **kwargs):
        raise ProcessInterrupted()

    monkeypatch.setattr(deploy.ReceiptStore, "write_yaml", stop)
    with pytest.raises(ProcessInterrupted):
        await deploy.deploy_file(e.path, yes=True)
    await deploy.cleanup_owned(e.store.read(), delete_function=True)
    assert not e.cloud.function_exists and not e.cloud.trigger_exists


@pytest.mark.anyio
async def test_initial_cleanup_keeps_resolution_evidence_after_partial_deletion(
    initial_cloud, monkeypatch
):
    e = initial_cloud

    def stop(*args, **kwargs):
        raise ProcessInterrupted()

    monkeypatch.setattr(deploy.ReceiptStore, "write_yaml", stop)
    with pytest.raises(ProcessInterrupted):
        await deploy.deploy_file(e.path, yes=True)
    e.cloud.fail = "delete_action"
    with pytest.raises(SandboxDeploymentError):
        await deploy.cleanup_owned(e.store.read(), delete_function=True)
    assert e.store.read().flow_id is None
    before = list(e.cloud.create_calls)
    with pytest.raises(SandboxDeploymentConflict, match="cleanup"):
        await deploy.deploy_file(e.path, yes=True)
    assert e.cloud.create_calls == before
    await deploy.cleanup_owned(e.store.read(), delete_function=True)
    assert e.store.read().function_id is None


@pytest.mark.anyio
async def test_deploy_after_completed_cleanup_is_a_new_create_intent(initial_cloud):
    e = initial_cloud
    await deploy.deploy_file(e.path, yes=True)
    original_keys = dict(e.cloud.create_calls)
    await deploy.cleanup_owned(e.store.read(), delete_function=True)
    result = await deploy.deploy_file(e.path, yes=True)
    assert result.flow_id == "flow" and e.cloud.flow["enabled"] is True
    new_keys = dict(e.cloud.create_calls)
    assert all(new_keys[kind] != key for kind, key in original_keys.items())
