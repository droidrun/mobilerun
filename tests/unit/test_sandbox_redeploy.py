"""Owned flow reconciliation must never silently adopt altered bindings."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.sandbox import deployment as deploy
from mobilerun.sandbox.errors import SandboxDeploymentConflict, SandboxDeploymentError
from mobilerun.sandbox.models import Artifact, CloudDeclaration, OwnershipReceipt
from mobilerun.sandbox.settings import SandboxSettings
from tests.support.sandbox_fakes import SandboxCloud


@pytest.fixture
def setup_deploy(tmp_path, monkeypatch):
    scope = deploy.Scope("https://example.test/v1", "a" * 64)
    settings = SandboxSettings(MOBILERUN_SANDBOX_DIRECTORY=str(tmp_path))
    art = Artifact(
        CloudDeclaration("demo", endpoint=True),
        "source",
        "digest",
        {"type": "object"},
        None,
    )
    receipt = OwnershipReceipt(
        scope.origin,
        scope.account_id,
        "demo",
        "f",
        "flow",
        "trigger",
        ("action",),
        "digest",
        (),
    )
    store = deploy.ReceiptStore(scope, "demo", settings)
    store.write(receipt)
    doc = deploy.build_document(art, "f", 1, device=None)
    fake = SimpleNamespace(
        list_functions=AsyncMock(
            return_value=[{"id": "f", "name": "demo", "latestVersion": {"version": 1}}]
        ),
        list_flows=AsyncMock(
            return_value=[
                {
                    "id": "flow",
                    "name": "demo · flow",
                    "triggerId": "trigger",
                    "enabled": False,
                }
            ]
        ),
        list_actions=AsyncMock(return_value=[{"actionId": "action"}]),
        get_document=AsyncMock(return_value=doc),
        get_version=AsyncMock(return_value={"version": 1, "status": "ready"}),
        get_source=AsyncMock(return_value="source"),
        disable_flow=AsyncMock(),
        create_version=AsyncMock(),
        enable_flow=AsyncMock(),
    )
    monkeypatch.setattr(deploy, "clients", fake)
    monkeypatch.setattr(deploy, "package", lambda path: art)
    monkeypatch.setattr(deploy, "current_scope", lambda: scope)
    monkeypatch.setattr(deploy, "get_sandbox_settings", lambda: settings)
    return SimpleNamespace(fake=fake, store=store, receipt=receipt, artifact=art)


@pytest.mark.anyio
@pytest.mark.parametrize("endpoint", [True, None])
async def test_unchanged_owned_disabled_flow_has_no_mutation(
    setup_deploy, monkeypatch, endpoint
):
    e = setup_deploy
    monkeypatch.setattr(
        deploy,
        "package",
        lambda path: replace(
            e.artifact, declaration=CloudDeclaration("demo", endpoint=endpoint)
        ),
    )
    result = await deploy.deploy_file(Path("fake.py"), yes=True)
    assert result.status == "unchanged" and result.flow_id == "flow"
    assert result.endpoint.trigger_id == "trigger"
    e.fake.disable_flow.assert_not_awaited()
    e.fake.enable_flow.assert_not_awaited()
    e.fake.create_version.assert_not_awaited()
    assert e.store.read() == e.receipt


@pytest.mark.anyio
async def test_altered_action_bindings_conflict_before_version_create(setup_deploy):
    e = setup_deploy
    e.fake.list_actions.return_value = [{"actionId": "foreign"}]
    e.fake.get_source.return_value = "old source"
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(Path("fake.py"), yes=True)
    e.fake.list_actions.assert_awaited_once()
    e.fake.create_version.assert_not_awaited()
    e.fake.disable_flow.assert_not_awaited()
    assert e.store.read() == e.receipt


@pytest.fixture
def changed(setup_deploy, monkeypatch):
    e = setup_deploy
    cloud = SandboxCloud(deploy.build_document(e.artifact, "f", 1, device=None))
    e.store.write_yaml(cloud.document)
    monkeypatch.setattr(deploy, "clients", cloud)
    monkeypatch.setattr(
        deploy,
        "package",
        lambda p: replace(
            e.artifact,
            source="changed",
            digest="changed",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        ),
    )
    e.cloud = cloud
    return e


@pytest.mark.anyio
async def test_changed_cutover(changed):
    e = changed
    result = await deploy.deploy_file(Path("fake.py"), device="phone", yes=True)
    assert result.version == 2
    assert e.cloud.flow["enabled"] is True
    assert e.cloud.calls.index("disable") < e.cloud.calls.index("create")
    assert e.cloud.calls.index("delete_action") < e.cloud.calls.index("enable")
    assert "action" not in e.cloud.actions
    assert e.store.read().action_ids == tuple(e.cloud.bindings)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "stage",
    ["disable", "create", "replace", "trigger", "devices", "delete_action", "enable"],
)
async def test_cutover_lost_response_recovery(changed, stage):
    e = changed
    e.cloud.fail = stage
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.deploy_file(Path("fake.py"), device="phone", yes=True)
    assert error.value.receipt.function_id == "f"
    assert e.store.read().stages
    if stage != "enable":
        assert "enable" not in e.cloud.calls
        assert e.cloud.flow["enabled"] is False
    keys = dict(e.cloud.keys)
    await deploy.deploy_file(Path("fake.py"), device="phone", yes=True)
    if keys:
        assert e.cloud.keys == keys
    assert e.cloud.flow["enabled"] is True


@pytest.mark.anyio
async def test_cleanup_retains_function_and_removes_yaml(changed):
    e = changed
    e.store.write_yaml(e.cloud.document)
    await deploy.cleanup_owned(e.receipt, billing="no_commerce")
    assert not e.store.yaml_path.exists()
    assert "delete_function" not in e.cloud.calls
    assert e.store.read().function_id == "f"


@pytest.mark.anyio
async def test_cleanup_continues_and_retains_failed_evidence(changed):
    e = changed
    e.cloud.fail = "delete_action"
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.cleanup_owned(e.receipt)
    assert error.value.receipt.action_ids == ("action",)
    assert "delete_trigger" in e.cloud.calls


@pytest.mark.anyio
async def test_uncertain_shutdown_never_creates_or_enables(changed, monkeypatch):
    e = changed

    async def no_shutdown(*args, **kwargs):
        return {"enabled": False}  # A successful response is not fresh proof.

    monkeypatch.setattr(e.cloud, "disable_flow", no_shutdown)
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.deploy_file(Path("fake.py"), yes=True)
    assert "enabled=True" in str(error.value)
    assert "create" not in e.cloud.calls and "enable" not in e.cloud.calls


@pytest.mark.anyio
async def test_divergent_remote_document_fails_before_mutation(changed):
    e = changed
    e.cloud.document["on"] = {"custom": {"inputs": {"foreign": True}}}
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(Path("fake.py"), yes=True)
    assert "disable" not in e.cloud.calls
    assert e.cloud.version == 1


@pytest.mark.anyio
async def test_pending_cutover_rejects_divergence(changed):
    e = changed
    e.cloud.fail = "replace"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(Path("fake.py"), yes=True)
    e.cloud.document["on"] = {"custom": {"foreign": True}}
    e.cloud.calls.clear()
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(Path("fake.py"), yes=True)
    assert "enable" not in e.cloud.calls and "disable" not in e.cloud.calls


@pytest.mark.anyio
async def test_deployers_conflict_on_identity_lock(changed):
    e = changed
    with e.store.lock():
        with pytest.raises(SandboxDeploymentConflict):
            await deploy.deploy_file(Path("fake.py"), yes=True)
    assert "disable" not in e.cloud.calls


@pytest.mark.anyio
async def test_verification_failure_retains_disabled_progress(changed, monkeypatch):
    e = changed
    original = e.cloud.get_document
    count = 0

    async def fail_verify(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("unavailable")
        return await original(*args, **kwargs)

    monkeypatch.setattr(e.cloud, "get_document", fail_verify)
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.deploy_file(Path("fake.py"), yes=True)
    assert error.value.receipt.action_ids == ("action", "new0")
    assert e.cloud.bindings == ["new0"]
    assert e.cloud.flow["enabled"] is False
    assert "enable" not in e.cloud.calls
    await deploy.deploy_file(Path("fake.py"), yes=True)
    assert e.cloud.flow["enabled"] is True


@pytest.mark.anyio
async def test_changed_disabled_flow_stays_disabled(changed):
    e = changed
    e.cloud.flow["enabled"] = False
    result = await deploy.deploy_file(Path("fake.py"), yes=True)
    assert e.cloud.flow["enabled"] is False
    assert "enable" not in e.cloud.calls
    assert result.warnings


@pytest.mark.anyio
async def test_cleanup_foreign_receipt_is_read_only(changed):
    e = changed
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.cleanup_owned(replace(e.receipt, action_ids=("foreign",)))
    assert e.cloud.calls == []


@pytest.mark.anyio
async def test_cleanup_failed_flow_deletion_retains_yaml(changed):
    e = changed
    e.cloud.fail = "delete_flow"
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.cleanup_owned(e.receipt)
    assert error.value.receipt.flow_id == "flow"
    assert e.store.yaml_path.exists()


@pytest.mark.anyio
async def test_cleanup_function_explicit_opt_in(changed):
    e = changed
    await deploy.cleanup_owned(e.receipt, delete_function=True)
    assert "delete_function" in e.cloud.calls
    assert e.store.read().function_id is None


@pytest.mark.anyio
async def test_action_intent_is_durable_before_lost_response(changed, monkeypatch):
    e = changed
    original = e.cloud.create_action
    keys = []

    async def create(params, key, **kwargs):
        pending = deploy._cutover(e.store.read())
        assert pending["pending"] == "create:0"
        assert any(key in stage for stage in e.store.read().stages)
        keys.append(key)
        return await original(params, key, **kwargs)

    monkeypatch.setattr(e.cloud, "create_action", create)
    e.cloud.fail = "create"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(Path("fake.py"), yes=True)
    await deploy.deploy_file(Path("fake.py"), yes=True)
    assert len(keys) == 2 and keys[0] == keys[1]


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["schedule", "device_removal"])
async def test_trigger_and_explicit_device_removal(changed, monkeypatch, kind):
    e = changed
    if kind == "schedule":
        declaration = CloudDeclaration("demo", schedule="0 * * * *")
    else:
        declaration = CloudDeclaration("demo")
    if kind == "device_removal":
        e.cloud.document["devices"] = ["old_phone"]
        e.store.write_yaml(e.cloud.document)
    monkeypatch.setattr(
        deploy, "package", lambda p: replace(e.artifact, declaration=declaration)
    )
    await deploy.deploy_file(Path("fake.py"), device_free=True, yes=True)
    assert e.cloud.document["on"] == (
        {"schedule": {"cron": "0 * * * *", "timezone": "UTC"}}
        if kind == "schedule"
        else {"custom": {"inputs": {"type": "object"}}}
    )
    assert not e.cloud.document.get("devices")


@pytest.mark.anyio
@pytest.mark.parametrize("shared", ["trigger", "action"])
async def test_shared_dependencies_reject_cutover_and_cleanup(
    changed, monkeypatch, shared
):
    e = changed
    original = e.cloud.list_flows

    async def flows(**kwargs):
        return [
            *(await original(**kwargs)),
            {
                "id": "foreign",
                "name": "foreign",
                "enabled": True,
                "triggerId": "trigger" if shared == "trigger" else "other",
            },
        ]

    monkeypatch.setattr(e.cloud, "list_flows", flows)
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(Path("fake.py"), yes=True)
    with pytest.raises(SandboxDeploymentError):
        await deploy.cleanup_owned(e.receipt)
    assert e.cloud.flow["enabled"] is True
    assert "delete_flow" not in e.cloud.calls
    if shared == "action":
        assert "action" in e.cloud.actions


@pytest.mark.anyio
async def test_two_failed_shutdown_retries_preserve_inflight_stage(
    changed, monkeypatch
):
    e = changed
    e.cloud.fail = "trigger"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(Path("fake.py"), yes=True)
    e.cloud.fail = "disable"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(Path("fake.py"), yes=True)
    await deploy.deploy_file(Path("fake.py"), yes=True)
    assert e.cloud.flow["enabled"] is True


@pytest.mark.anyio
async def test_cleanup_pending_unknown_action_reports_uncertainty(changed):
    e = changed
    e.cloud.fail = "create"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(Path("fake.py"), yes=True)
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.cleanup_owned(e.store.read(), delete_function=True)
    assert "creation outcome uncertain" in str(error.value)
    assert e.store.read().function_id == "f"
    assert "delete_function" not in e.cloud.calls


@pytest.mark.anyio
async def test_cleanup_lost_flow_delete_resolves_on_retry(changed):
    e = changed
    e.cloud.fail = "delete_flow"
    with pytest.raises(SandboxDeploymentError):
        await deploy.cleanup_owned(e.receipt)
    assert e.store.yaml_path.exists()
    await deploy.cleanup_owned(e.store.read())
    assert not e.store.yaml_path.exists()
    assert e.store.read().flow_id is None


@pytest.mark.anyio
async def test_uncertain_enable_attempts_shutdown_and_reports_actual_state(changed):
    e = changed
    e.cloud.fail = "enable"
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.deploy_file(Path("fake.py"), yes=True)
    assert e.cloud.flow["enabled"] is False
    assert "enabled=False" in str(error.value)
    assert deploy._cutover(error.value.receipt)["warning"] == str(error.value)
    assert e.cloud.calls[-1] == "disable"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "stage", ["delete_action", "delete_trigger", "delete_function"]
)
async def test_cleanup_lost_delete_response_real_404_completes_on_retry(changed, stage):
    e = changed
    e.cloud.fail = stage
    with pytest.raises(SandboxDeploymentError):
        await deploy.cleanup_owned(e.receipt, delete_function=True)
    await deploy.cleanup_owned(e.store.read(), delete_function=True)
    receipt = e.store.read()
    assert (
        receipt.action_ids == ()
        and receipt.trigger_id is None
        and receipt.function_id is None
    )


@pytest.mark.anyio
async def test_fake_missing_delete_is_not_silently_idempotent(changed):
    from mobilerun.errors.cloud import CloudResourceNotFound

    await changed.cloud.delete_action("action")
    with pytest.raises(CloudResourceNotFound):
        await changed.cloud.delete_action("action")
