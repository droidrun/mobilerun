"""Sandbox deployment documents, first deploy, cutover, and resource cleanup."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from mobilerun_functions.schema import output_schema
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from mobilerun.core.webhooks import WebhookEndpoint
from mobilerun.errors.cloud import BillingRequired, CloudResourceNotFound
from mobilerun.sandbox import cleanup as sandbox_cleanup
from mobilerun.sandbox import deployment as deploy
from mobilerun.sandbox import documents as sandbox_documents
from mobilerun.sandbox import flows as sandbox_flows
from mobilerun.sandbox import receipts as sandbox_receipts
from mobilerun.sandbox.errors import (
    FlowCapabilityError,
    FlowStagingError,
    SandboxDeclarationError,
    SandboxDeploymentConflict,
    SandboxDeploymentError,
)
from mobilerun.sandbox.models import Artifact, CloudDeclaration, OwnershipReceipt
from tests.support.sandbox_fakes import ProcessInterrupted
from tests.support.sandbox_fixtures import setup_deploy as setup_deploy
from tests.unit.sandbox.staging_fixtures import base_changed as base_changed
from tests.unit.sandbox.staging_fixtures import (
    base_env as base_env,
)
from tests.unit.sandbox.staging_fixtures import changed as changed
from tests.unit.sandbox.staging_fixtures import (
    env as env,
)
from tests.unit.sandbox.staging_fixtures import (
    initial_cloud as initial_cloud,
)


def artifact(**options):
    return Artifact(
        CloudDeclaration("hello", **options),
        "source",
        "digest",
        {"type": "object"},
        None,
    )


def test_build_endpoint_document_uses_pinned_version_and_no_device():
    from mobilerun.sandbox.documents import build_document

    doc = build_document(artifact(endpoint=True), "function", 3, device=None)
    assert doc["on"] == {"custom": {"inputs": {"type": "object"}}}
    assert "devices" not in doc
    assert doc["steps"][0]["with"]["version"] == 3


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ("", {"custom": {"inputs": {"type": "object"}}}),
        (", endpoint=False", None),
        (
            ", schedule='0 * * * *'",
            {"schedule": {"cron": "0 * * * *", "timezone": "UTC"}},
        ),
    ],
)
def test_literal_declaration_flow_resolution(tmp_path, options, expected):
    from mobilerun.sandbox.documents import build_document
    from mobilerun.sandbox.packaging import read_declaration

    path = tmp_path / "main.py"
    path.write_text(
        "from mobilerun.sandbox import cloud\n"
        f"@cloud('hello'{options})\ndef hello(): pass\n"
    )
    packaged = Artifact(
        read_declaration(path), "source", "digest", {"type": "object"}, None
    )
    doc = build_document(packaged, "function", 1, device=None)
    assert (doc["on"] if doc is not None else None) == expected


@pytest.mark.parametrize(
    "options",
    [
        "schedule='0 * * * *', endpoint=True",
        "endpoint=1",
    ],
)
def test_literal_declaration_rejects_invalid_trigger_options(tmp_path, options):
    from mobilerun.sandbox.errors import SandboxDeclarationError
    from mobilerun.sandbox.packaging import read_declaration

    path = tmp_path / "main.py"
    path.write_text(
        "from mobilerun.sandbox import cloud\n"
        f"@cloud('hello', {options})\ndef hello(): pass\n"
    )
    with pytest.raises(SandboxDeclarationError):
        read_declaration(path)


def test_literal_declaration_rejects_removed_manual_trigger(tmp_path):
    from mobilerun.sandbox.errors import SandboxDeclarationError
    from mobilerun.sandbox.packaging import read_declaration

    path = tmp_path / "main.py"
    path.write_text(
        "from mobilerun.sandbox import cloud\n"
        "@cloud('hello', trigger='manual')\ndef hello(): pass\n"
    )
    with pytest.raises(
        SandboxDeclarationError, match=r"manual.*removed.*endpoint.*device="
    ):
        read_declaration(path)


def test_invalid_cron_fails_locally():
    from mobilerun.sandbox.documents import build_document
    from mobilerun.sandbox.errors import SandboxDeclarationError

    with pytest.raises(SandboxDeclarationError):
        build_document(artifact(schedule="bad"), "function", 1, device=None)


@pytest.mark.anyio
async def test_webhook_file_list_must_be_fixed_length_before_deploy(env, hooks):
    class Event(BaseModel):
        image: str

    class Output(BaseModel):
        events: list[Event] = Field(min_length=1, max_length=3)

    decl = CloudDeclaration(
        "demo", webhook=HOOK_ID, files={"map.jpg": "events.0.image"}, files_folder="atc"
    )
    packaged = replace(
        env.artifact, declaration=decl, output_schema=output_schema(Output)
    )
    with pytest.raises(
        SandboxDeclarationError, match=r"fixed.length.*min_length.*max_length"
    ):
        await deploy.deploy_file(env.path, artifact=packaged)
    env.fake.create_function.assert_not_awaited()
    hooks.get.assert_not_awaited()


def test_webhook_keeps_fixed_list_items_without_files_as_whole_values():
    class Event(BaseModel):
        image: str
        title: str

    class Output(BaseModel):
        events: list[Event] = Field(min_length=2, max_length=2)

    packaged = replace(
        artifact(
            webhook=HOOK_ID, files={"map.jpg": "events.0.image"}, files_folder="atc"
        ),
        output_schema=output_schema(Output),
    )
    doc = sandbox_documents.build_document(packaged, "f", 1, device=None)
    assert doc["steps"][-1]["with"]["data"] == {
        "result": {
            "events": [
                {
                    "image": {
                        "path": "/atc/map.jpg",
                        "fileId": "{{steps.upload_1.body.fileId}}",
                    },
                    "title": "{{steps.function.body.result.events.0.title}}",
                },
                "{{steps.function.body.result.events.1}}",
            ]
        }
    }


@pytest.mark.anyio
@pytest.mark.parametrize("shape", ["extra", "exclude_if"])
async def test_webhook_rejects_expanded_objects_with_unstable_keys(env, hooks, shape):
    if shape == "extra":

        class Event(BaseModel):
            model_config = ConfigDict(extra="allow")
            image: str

    else:

        class Event(BaseModel):
            model_config = ConfigDict(json_schema_serialization_defaults_required=True)
            image: str
            note: str = Field(default="", exclude_if=lambda note: not note)

    class Output(BaseModel):
        event: Event

    declaration = CloudDeclaration(
        "demo", webhook=HOOK_ID, files={"map.jpg": "event.image"}, files_folder="atc"
    )
    packaged = replace(
        env.artifact, declaration=declaration, output_schema=output_schema(Output)
    )
    with pytest.raises(SandboxDeclarationError, match="Webhook result objects"):
        await deploy.deploy_file(env.path, artifact=packaged)
    env.fake.create_function.assert_not_awaited()
    hooks.get.assert_not_awaited()


def test_webhook_file_path_normalizes_folder_without_changing_filename():
    class Output(BaseModel):
        image: str

    filename = "re\u0301sume\u0301.jpg"
    packaged = replace(
        artifact(
            webhook=HOOK_ID, files={filename: "image"}, files_folder="atc/cafe\u0301"
        ),
        output_schema=output_schema(Output),
    )
    doc = sandbox_documents.build_document(packaged, "f", 1, device=None)
    assert doc["steps"][1]["with"]["filename"] == filename
    assert (
        doc["steps"][-1]["with"]["data"]["result"]["image"]["path"]
        == f"/atc/café/{filename}"
    )


def test_webhook_uses_first_upload_when_a_field_has_multiple_filenames():
    class Output(BaseModel):
        image: str
        other: str

    packaged = replace(
        artifact(
            webhook=HOOK_ID,
            files={"other.jpg": "other", "first.jpg": "image", "second.png": "image"},
            files_folder="atc",
        ),
        output_schema=output_schema(Output),
    )
    doc = sandbox_documents.build_document(packaged, "f", 1, device=None)
    assert [step["with"]["filename"] for step in doc["steps"][1:-1]] == [
        "other.jpg",
        "first.jpg",
        "second.png",
    ]
    assert doc["steps"][-1]["with"]["data"]["result"]["image"] == {
        "path": "/atc/first.jpg",
        "fileId": "{{steps.upload_2.body.fileId}}",
    }


def test_legacy_file_webhook_order_can_only_be_read_for_redeploy():
    document = {
        "steps": [
            {
                "id": "function",
                "uses": "functions.run",
                "with": {"functionId": "f", "version": 7},
            },
            {"id": "webhook", "uses": "webhooks.send"},
            {"id": "upload_1", "uses": "files.upload"},
        ]
    }
    with pytest.raises(SandboxDeploymentConflict, match="redeploy"):
        sandbox_flows.pinned_version("f", document)
    assert sandbox_flows.pinned_version("f", document, for_redeploy=True) == 7


@pytest.mark.parametrize(
    ("filename", "folder"),
    [
        ("{{steps.function.body.result.image}}.jpg", "atc"),
        ("map.jpg", "atc/{{meta.shortDate}}"),
    ],
)
def test_webhook_file_path_cannot_embed_a_template(filename, folder):
    class Output(BaseModel):
        image: str

    packaged = replace(
        artifact(
            webhook=HOOK_ID,
            files={filename: "image"},
            files_folder=folder,
        ),
        output_schema=output_schema(Output),
    )
    with pytest.raises(SandboxDeclarationError, match="template syntax"):
        sandbox_documents.build_document(packaged, "f", 1, device=None)


@pytest.mark.parametrize(
    "declaration", [CloudDeclaration(""), CloudDeclaration("x", endpoint="yes")]
)
def test_invalid_declaration_is_rejected(declaration):
    from mobilerun.sandbox.documents import build_document
    from mobilerun.sandbox.errors import SandboxDeclarationError

    with pytest.raises(SandboxDeclarationError):
        build_document(
            Artifact(declaration, "source", "digest", {}, None), "f", 1, device=None
        )


def test_equivalence_detects_device_removal_and_schema_changes():
    from mobilerun.sandbox.documents import equivalent_document

    assert not equivalent_document(
        {"on": {"custom": {}}, "devices": ["phone"]}, {"on": {"custom": {}}}
    )
    assert not equivalent_document(
        {"on": {"custom": {"inputs": {}}}}, {"on": {"custom": {}}}
    )
    assert equivalent_document({"name": "x", "server": 1}, {"name": "x"})


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
    for billing in ("no_commerce", "included"):
        with pytest.raises(BillingRequired, match="--yes"):
            await deploy.deploy_file(env.path, billing=billing)
    env.fake.list_functions.assert_not_awaited()
    assert not env.store.directory.exists()


@pytest.mark.anyio
@pytest.mark.parametrize("billing", [None, "paid"])
async def test_paid_deployment_skips_confirmation(env, monkeypatch, billing):
    monkeypatch.setenv(
        "MOBILERUN_BILLING", "paid" if billing is None else "no_commerce"
    )
    kwargs = {} if billing is None else {"billing": billing}
    assert (await deploy.deploy_file(env.path, **kwargs)).status == "created"
    env.fake.create_document.assert_awaited_once()


def test_sdk_declined_deploy_writes_no_receipt_and_makes_no_cloud_calls(
    env, monkeypatch
):
    from functools import partial

    from mobilerun.clients.cloud.billing import with_billing_retry
    from mobilerun.sandbox import app

    monkeypatch.setattr(app, "package", lambda _: env.artifact)
    prompt = AsyncMock(return_value=False)
    monkeypatch.setattr(
        app,
        "with_billing_retry",
        partial(with_billing_retry, prompt=prompt, is_interactive=lambda: True),
    )

    with pytest.raises(BillingRequired):
        app.deploy(env.path, billing="included")

    prompt.assert_awaited_once()
    assert not env.store.directory.exists()
    assert all(not fake.called for fake in vars(env.fake).values())


@pytest.mark.anyio
async def test_device_selection_and_invalid_schema_precede_cloud(env, monkeypatch):
    schedule = CloudDeclaration("demo", schedule="* * * * *")
    monkeypatch.setattr(
        deploy, "package", lambda path: replace(env.artifact, declaration=schedule)
    )
    with pytest.raises(SandboxDeclarationError):
        await deploy.deploy_file(env.path)
    monkeypatch.setattr(
        deploy,
        "package",
        lambda path: replace(env.artifact, input_schema={"$ref": "#/$defs/X"}),
    )
    with pytest.raises(SandboxDeclarationError):
        await deploy.deploy_file(env.path)
    env.fake.list_functions.assert_not_awaited()


@pytest.mark.anyio
async def test_duplicate_and_foreign_function_rejected(env):
    env.fake.list_functions.return_value = [
        {"id": "f", "name": "demo"},
        {"id": "g", "name": "demo"},
    ]
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(env.path)
    env.fake.list_functions.return_value = [{"id": "f", "name": "demo"}]
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(env.path)
    env.fake.create_function.assert_not_awaited()
    env.fake.create_document.assert_not_awaited()


HOOK_ID = "abcdef00-0000-4000-8000-000000000001"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "target", ["https://example.test/?token=sentinel", "not-an-id", 42]
)
async def test_invalid_webhook_prevents_all_cloud_calls(env, hooks, target):
    with pytest.raises(SandboxDeclarationError) as error:
        await deploy.deploy_file(
            env.path,
            artifact=replace(
                env.artifact, declaration=CloudDeclaration("demo", webhook=target)
            ),
        )
    assert (
        str(error.value)
        == "webhook= takes a webhook endpoint ID. Register one first: mobilerun webhook register <url>"
    )
    assert not hooks.mock_calls
    for method in vars(env.fake).values():
        method.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("deleted", [True, False])
async def test_initial_retry_rechecks_endpoint_before_new_staging(env, hooks, deleted):
    artifact = replace(
        env.artifact, declaration=CloudDeclaration("demo", webhook=HOOK_ID)
    )
    env.fake.validate_document.return_value = {"valid": False}
    with pytest.raises(SandboxDeploymentError, match="validation failed"):
        await deploy.deploy_file(env.path, artifact=artifact)

    env.fake.validate_document.return_value = {"valid": True}
    env.fake.create_document.return_value["actionIds"].append("send")
    if deleted:
        hooks.get.side_effect = CloudResourceNotFound(status=404)
    else:
        hooks.get.side_effect = lambda id: endpoint(id, health="blocked")
    with pytest.raises(
        SandboxDeploymentError, match="not found" if deleted else "blocked"
    ):
        await deploy.deploy_file(env.path, artifact=artifact)
    env.fake.prepare_action_payloads.assert_not_awaited()
    env.fake.create_document.assert_not_awaited()


@pytest.mark.anyio
async def test_staging_replays_before_rechecking_deleted_borrowed_endpoint(env, hooks):
    artifact = replace(
        env.artifact,
        declaration=CloudDeclaration("demo", webhook=HOOK_ID),
    )
    staged = dict(flowId="flow", triggerId="trigger", actionIds=["action", "send"])
    env.fake.create_document.side_effect = [
        FlowStagingError("trigger", ["action"], None),
        staged,
    ]
    with pytest.raises(SandboxDeploymentError, match="staging failed"):
        await deploy.deploy_file(env.path, artifact=artifact)
    hooks.get.side_effect = CloudResourceNotFound(status=404)
    with pytest.raises(SandboxDeploymentError, match="not found"):
        await deploy.deploy_file(env.path, artifact=artifact)
    assert (
        env.fake.create_document.await_args_list[0]
        == env.fake.create_document.await_args_list[1]
    )
    env.fake.prepare_action_payloads.assert_awaited_once()
    env.fake.enable_flow.assert_not_awaited()
    receipt = env.store.read()
    assert (
        receipt.flow_id == "flow"
        and receipt.trigger_id == "trigger"
        and receipt.action_ids == ("action", "send")
    )
    env.fake.delete_action = AsyncMock()
    env.fake.delete_trigger = AsyncMock()
    await sandbox_cleanup.cleanup_owned(receipt)
    assert not env.store.read().action_ids


def endpoint(endpoint_id=HOOK_ID, **fields):
    return replace(
        WebhookEndpoint(
            endpoint_id,
            "http",
            SecretStr("https://receiver.example/private-token"),
            "ACTIVE",
            "healthy",
            (),
            SecretStr("signing-sentinel"),
        ),
        **fields,
    )


@pytest.mark.anyio
async def test_paused_initial_webhook_and_explicit_enable(env, hooks):
    art = replace(
        env.artifact,
        declaration=CloudDeclaration("demo", webhook=HOOK_ID),
    )
    fake = env.fake
    fake.create_document.return_value["actionIds"].append("send")
    fake.enable_flow.side_effect = RuntimeError()
    with pytest.raises(SandboxDeploymentError, match="activation"):
        await deploy.deploy_file(env.path, artifact=art)
    fake.list_functions.return_value = [
        fake.create_function.return_value | {"name": "demo"}
    ]
    fake.list_flows.return_value = [
        {"id": "flow", "name": "demo · flow", "triggerId": "trigger", "enabled": False}
    ]
    fake.list_actions = AsyncMock(
        return_value=[{"actionId": "action"}, {"actionId": "send"}]
    )
    fake.get_document = AsyncMock(return_value=fake.create_document.await_args.args[0])
    fake.disable_flow = AsyncMock()
    await sandbox_flows.set_enabled("demo", False)
    hooks.reset_mock()
    await deploy.deploy_file(env.path, artifact=art)
    hooks.get.assert_not_awaited()
    fake.enable_flow.reset_mock()
    hooks.get.side_effect = lambda id: endpoint(id, health="blocked")
    with pytest.raises(SandboxDeploymentError, match="blocked"):
        await sandbox_flows.set_enabled("demo", True)
    hooks.get.assert_awaited_once_with(HOOK_ID)
    fake.enable_flow.assert_not_awaited()


@pytest.fixture
def hooks(monkeypatch):
    manager = AsyncMock()
    manager.get.side_effect = lambda id: endpoint(id)
    monkeypatch.setattr(sandbox_flows, "CloudWebhooks", lambda **kwargs: manager)
    return manager


@pytest.mark.anyio
async def test_existing_endpoint_id_stages_result_step(env, hooks):
    env.fake.create_document.return_value["actionIds"].append("send")
    await deploy.deploy_file(
        env.path,
        artifact=replace(
            env.artifact, declaration=CloudDeclaration("demo", webhook=HOOK_ID)
        ),
    )
    doc = env.fake.create_document.await_args.args[0]
    assert doc["steps"][-1]["uses"] == "webhooks.send"
    assert doc["steps"][-1]["with"] == {
        "endpointId": HOOK_ID,
        "data": {"result": "{{steps.function.body.result}}"},
    }
    assert [call.args for call in hooks.get.await_args_list] == [(HOOK_ID,), (HOOK_ID,)]


@pytest.mark.anyio
async def test_unknown_webhook_id_has_registration_hint(env, hooks):
    hooks.get.side_effect = CloudResourceNotFound(status=404)
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.deploy_file(
            env.path,
            artifact=replace(
                env.artifact, declaration=CloudDeclaration("demo", webhook=HOOK_ID)
            ),
        )
    assert (
        str(error.value)
        == f"Webhook endpoint {HOOK_ID} not found. Register one: mobilerun webhook register <url>"
    )
    env.fake.create_function.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"state": "DISABLED"}, "disabled"),
        ({"health": "blocked"}, "blocked"),
        ({"event_types": ("other",)}, "workflow.step.emit"),
    ],
)
async def test_reused_webhook_must_be_usable(env, hooks, fields, message):
    hooks.get.side_effect = lambda id: endpoint(id, **fields)
    with pytest.raises(SandboxDeploymentError, match=message) as error:
        await deploy.deploy_file(
            env.path,
            artifact=replace(
                env.artifact,
                declaration=CloudDeclaration("demo", webhook=HOOK_ID),
            ),
        )
    assert "mobilerun webhook register <url>" in str(error.value)
    env.fake.create_function.assert_not_awaited()


@pytest.mark.anyio
async def test_new_endpoint_stages_persists_and_enables(env):
    result = await deploy.deploy_file(env.path)
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
        await deploy.deploy_file(env.path)
    assert error.value.receipt.trigger_id == "trigger"
    assert env.store.read().action_ids == ("action",)
    assert not env.store.yaml_path.exists()
    env.fake.enable_flow.assert_not_awaited()


@pytest.mark.anyio
async def test_identical_ready_reuses_version(env):
    owned(env)
    result = await deploy.deploy_file(env.path)
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
        await deploy.deploy_file(env.path)
    env.fake.enable_flow.assert_not_awaited()


@pytest.mark.anyio
async def test_changed_source_creates_and_polls_exact_version(env):
    owned(env)
    env.fake.get_source.return_value = "old source"
    env.fake.get_version.side_effect = [
        {"version": 1, "status": "ready"},
        {"version": 2, "status": "ready"},
    ]
    result = await deploy.deploy_file(env.path)
    assert (result.status, result.version) == ("pushed", 2)
    assert env.fake.get_version.await_args_list[-1].args == ("f", 2)


@pytest.mark.anyio
async def test_enable_failure_exposes_persisted_receipt(env):
    env.fake.enable_flow.side_effect = RuntimeError("sensitive upstream detail")
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.deploy_file(env.path)
    assert error.value.receipt.flow_id == "flow"
    assert "sensitive" not in str(error.value)
    assert env.store.read().flow_id == "flow"


@pytest.mark.anyio
async def test_pending_inspection_eventually_ready(env):
    env.fake.get_version.side_effect = [
        {"version": 1, "status": "pending_inspection"},
        {"version": 1, "status": "ready"},
    ]
    result = await deploy.deploy_file(env.path)
    assert result.version == 1


@pytest.mark.anyio
async def test_malformed_staging_response_keeps_known_ids(env):
    env.fake.create_document.return_value = {"flowId": "flow", "triggerId": "trigger"}
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(env.path)
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
    result = await deploy.deploy_file(env.path, device_free=True)
    doc = env.fake.create_document.await_args.args[0]
    assert "devices" not in doc
    assert result.endpoint is None
    env.fake.enable_flow.assert_awaited_once()


@pytest.mark.anyio
async def test_flow_validation_failure_does_not_stage(env):
    env.fake.validate_document.return_value = {"valid": False}
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.deploy_file(env.path)
    assert error.value.receipt.function_id == "f"
    env.fake.create_document.assert_not_awaited()
    assert not env.store.yaml_path.exists()


@pytest.mark.anyio
async def test_unowned_flow_blocks_before_creating_function(env):
    env.fake.list_flows.return_value = [{"id": "other", "name": "demo · flow"}]
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(env.path)
    env.fake.create_function.assert_not_awaited()
    assert not env.store.yaml_path.exists()


@pytest.mark.anyio
async def test_unconfirmed_enable_is_not_reported_as_success(env):
    env.fake.enable_flow.return_value = {"id": "flow", "enabled": False}
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.deploy_file(env.path)
    assert error.value.receipt.flow_id == "flow"


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
        await deploy.deploy_file(e.path)
    assert e.store.read().stages
    first_keys = dict(e.cloud.create_calls)
    result = await deploy.deploy_file(e.path)
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
        await deploy.deploy_file(e.path)
    receipt = e.store.read()
    assert receipt.trigger_id == "trigger" and not receipt.action_ids
    with pytest.raises(SandboxDeploymentError, match="creation.*uncertain"):
        await sandbox_cleanup.cleanup_owned(receipt, delete_function=True)
    assert e.cloud.function_exists
    assert "delete_function" not in e.cloud.calls


@pytest.mark.anyio
async def test_pending_initial_intent_cannot_be_changed(initial_cloud, monkeypatch):
    e = initial_cloud
    e.cloud.fail = "flow"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(e.path)
    before = list(e.cloud.create_calls)
    monkeypatch.setattr(
        deploy,
        "package",
        lambda p: replace(e.artifact, source="different", digest="different"),
    )
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(e.path)
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
    result = await deploy.deploy_file(env.path)
    assert result.flow_id == "flow"


@pytest.mark.anyio
async def test_initial_action_replays_payload_after_catalog_changes(initial_cloud):
    e = initial_cloud
    sdk = e.cloud.sdk.workflows
    sdk.actions.create = AsyncMock(wraps=sdk.actions.create)
    e.cloud.fail = "action"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(e.path)
    first = sdk.actions.create.await_args.kwargs
    payload = {k: v for k, v in first.items() if k != "idempotency_key"}
    journal = sandbox_receipts.pending_initial(e.store.read())
    assert journal["action_payloads"] == [payload]
    sdk.action_catalog.list = AsyncMock(
        return_value={
            "data": {
                "items": [{"id": "changed", "stepName": "functions.run"}],
                "pagination": {"hasNext": False},
            }
        }
    )
    await deploy.deploy_file(e.path)
    assert sdk.actions.create.await_count == 2
    assert sdk.actions.create.await_args.kwargs == first
    sdk.action_catalog.list.assert_not_awaited()
    await sandbox_cleanup.cleanup_owned(e.store.read())
    assert not e.store.read().action_ids


@pytest.mark.anyio
async def test_interrupted_receipt_export_cannot_activate_until_replayed(
    initial_cloud, monkeypatch
):
    e = initial_cloud
    original = sandbox_receipts.ReceiptStore.write_yaml

    def stop(*args, **kwargs):
        raise ProcessInterrupted()

    monkeypatch.setattr(sandbox_receipts.ReceiptStore, "write_yaml", stop)
    with pytest.raises(ProcessInterrupted):
        await deploy.deploy_file(e.path)
    assert e.store.read().flow_id == "flow" and e.store.read().action_ids == ("action",)
    with pytest.raises(SandboxDeploymentConflict):
        await sandbox_flows.set_enabled("demo", True)
    assert "enable" not in e.cloud.calls
    monkeypatch.setattr(sandbox_receipts.ReceiptStore, "write_yaml", original)
    await deploy.deploy_file(e.path)
    assert e.cloud.flow["enabled"] is True


@pytest.mark.anyio
@pytest.mark.parametrize("resource", ["function", "flow"])
async def test_pending_create_does_not_adopt_conflicting_visible_name(
    initial_cloud, monkeypatch, resource
):
    e = initial_cloud
    e.cloud.fail = resource
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(e.path)
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
        await deploy.deploy_file(e.path)
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

    monkeypatch.setattr(sandbox_receipts.ReceiptStore, "write_yaml", stop)
    with pytest.raises(ProcessInterrupted):
        await deploy.deploy_file(e.path)
    await sandbox_cleanup.cleanup_owned(e.store.read(), delete_function=True)
    assert not e.cloud.function_exists and not e.cloud.trigger_exists


@pytest.mark.anyio
async def test_initial_cleanup_keeps_resolution_evidence_after_partial_deletion(
    initial_cloud, monkeypatch
):
    e = initial_cloud

    def stop(*args, **kwargs):
        raise ProcessInterrupted()

    monkeypatch.setattr(sandbox_receipts.ReceiptStore, "write_yaml", stop)
    with pytest.raises(ProcessInterrupted):
        await deploy.deploy_file(e.path)
    e.cloud.fail = "delete_action"
    with pytest.raises(SandboxDeploymentError):
        await sandbox_cleanup.cleanup_owned(e.store.read(), delete_function=True)
    assert e.store.read().flow_id is None
    before = list(e.cloud.create_calls)
    with pytest.raises(SandboxDeploymentConflict, match="cleanup"):
        await deploy.deploy_file(e.path)
    assert e.cloud.create_calls == before
    await sandbox_cleanup.cleanup_owned(e.store.read(), delete_function=True)
    assert e.store.read().function_id is None


@pytest.mark.anyio
async def test_deploy_after_completed_cleanup_is_a_new_create_intent(initial_cloud):
    e = initial_cloud
    await deploy.deploy_file(e.path)
    original_keys = dict(e.cloud.create_calls)
    await sandbox_cleanup.cleanup_owned(e.store.read(), delete_function=True)
    result = await deploy.deploy_file(e.path)
    assert result.flow_id == "flow" and e.cloud.flow["enabled"] is True
    new_keys = dict(e.cloud.create_calls)
    assert all(new_keys[kind] != key for kind, key in original_keys.items())


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
    result = await deploy.deploy_file(Path("fake.py"))
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
        await deploy.deploy_file(Path("fake.py"))
    e.fake.list_actions.assert_awaited_once()
    e.fake.create_version.assert_not_awaited()
    e.fake.disable_flow.assert_not_awaited()
    assert e.store.read() == e.receipt


@pytest.fixture
def webhook_cutover(changed):
    cloud = changed.cloud
    replace_actions = cloud.replace_flow_actions

    async def replace_typed(flow_id, refs, key, **kwargs):
        await replace_actions(flow_id, refs, key, **kwargs)
        for step, ref in zip(cloud.document["steps"], refs, strict=True):
            if (
                cloud.actions[ref["action_id"]]["catalog_entry_id"]
                == "webhooks-catalog"
            ):
                step["uses"] = "webhooks.send"
            cloud.actions[ref["action_id"]]["uses"] = step["uses"]

    cloud.replace_flow_actions = replace_typed

    async def deploy_to(url):
        artifact = replace(
            changed.artifact, declaration=CloudDeclaration("demo", webhook=url)
        )
        return await deploy.deploy_file(Path("fake.py"), artifact=artifact)

    return deploy_to


@pytest.mark.anyio
async def test_catalog_failure_leaves_old_flow_enabled(changed, hooks, webhook_cutover):
    cloud = changed.cloud
    failure = FlowCapabilityError(
        operation="Resolve webhook action", code="request_failed"
    )
    cloud.prepare_action_payloads.side_effect = failure
    flow, bindings, actions, document = deepcopy(
        (cloud.flow, cloud.bindings, cloud.actions, cloud.document)
    )
    with pytest.raises(FlowCapabilityError):
        await webhook_cutover(HOOK_ID)
    assert cloud.flow == flow and cloud.flow["enabled"] is True
    assert (
        cloud.bindings == bindings
        and cloud.actions == actions
        and cloud.document == document
    )
    assert cloud.calls == ["read_document"]


@pytest.mark.anyio
async def test_cutover_recovers_lost_webhook_action_before_activation(
    changed, hooks, webhook_cutover
):
    cloud = changed.cloud
    create = cloud.create_action
    requests = []

    async def lose_response(payload, key, **kwargs):
        result = await create(payload, key, **kwargs)
        if payload["catalog_entry_id"] == "webhooks-catalog":
            requests.append((payload, key))
            if len(requests) == 1:
                raise TimeoutError
        return result

    cloud.create_action = lose_response
    with pytest.raises(SandboxDeploymentError):
        await webhook_cutover(HOOK_ID)
    hooks.get.side_effect = CloudResourceNotFound(status=404)
    hooks.get.reset_mock()
    cloud.prepare_action_payloads.side_effect = AssertionError(
        "must replay the stored payload"
    )
    with pytest.raises(
        SandboxDeploymentError, match="mobilerun webhook register <url>"
    ):
        await webhook_cutover(HOOK_ID)
    assert len(requests) == 2 and requests[0] == requests[1]
    hooks.get.assert_awaited_once_with(HOOK_ID)
    assert cloud.keys[requests[0][1]] in changed.store.read().action_ids
    assert cloud.flow["enabled"] is True and cloud.bindings == ["action"]
    await sandbox_cleanup.cleanup_owned(changed.store.read())
    assert not cloud.actions


@pytest.mark.anyio
@pytest.mark.parametrize("check", [2, 3], ids=["before-replace", "before-enable"])
@pytest.mark.parametrize("retarget", [True, False], ids=["new-id", "remove"])
async def test_deleted_webhook_can_be_retargeted_in_place(
    changed, hooks, webhook_cutover, check, retarget
):
    cloud = changed.cloud
    old_ids = tuple(cloud.bindings)
    new_endpoint = "abcdef00-0000-4000-8000-000000000002"

    async def get_endpoint(endpoint_id):
        if endpoint_id == HOOK_ID and hooks.get.await_count >= check:
            raise CloudResourceNotFound(status=404)
        return endpoint(endpoint_id)

    hooks.get.side_effect = get_endpoint
    with pytest.raises(SandboxDeploymentError, match="not found"):
        await webhook_cutover(HOOK_ID)
    if check == 2:
        assert tuple(cloud.bindings) == old_ids
        assert cloud.flow["enabled"] is True
    else:
        assert tuple(cloud.bindings) != old_ids
    assert all(action_id in cloud.actions for action_id in old_ids)

    result = await webhook_cutover(new_endpoint if retarget else None)
    assert result.flow_id == "flow" and cloud.flow["enabled"] is True
    if retarget:
        assert cloud.document["steps"][-1]["with"]["endpointId"] == new_endpoint
    else:
        assert [step["uses"] for step in cloud.document["steps"]] == ["functions.run"]
    assert set(cloud.actions) == set(cloud.bindings) == set(result.action_ids)


@pytest.mark.anyio
async def test_redeploy_add_change_remove_endpoint_id(changed, hooks, webhook_cutover):
    second = "abcdef00-0000-4000-8000-000000000002"
    for target in (HOOK_ID, second, None):
        result = await webhook_cutover(target)
        assert (
            result.function_id,
            result.version,
            result.flow_id,
            result.trigger_id,
        ) == ("f", 1, "flow", "trigger")
        steps = changed.cloud.document["steps"]
        if target:
            assert steps[-1]["with"]["endpointId"] == target
        else:
            assert len(steps) == 1
        if target:
            changed.store.yaml_path.unlink()
        assert (await webhook_cutover(target)).status == "unchanged"
        assert changed.store.yaml_path.exists()
    await sandbox_cleanup.cleanup_owned(changed.store.read())
    assert all(call[0] == "get" for call in hooks.mock_calls)


@pytest.mark.anyio
async def test_redeploy_add_move_remove_file_delivery(changed, hooks):
    cloud = changed.cloud
    replace_actions = cloud.replace_flow_actions

    async def replace_typed(flow_id, refs, key, **kwargs):
        uses = [cloud.actions[ref["action_id"]]["uses"] for ref in refs]
        if "webhooks.send" in uses:
            producer = cloud.actions[refs[0]["action_id"]]
            assert producer["name"] == "function"
            assert uses[-1] == "webhooks.send"
        assert not cloud.document["settings"].get("delivery") or "files.upload" in uses
        await replace_actions(flow_id, refs, key, **kwargs)
        for step, ref in zip(cloud.document["steps"], refs, strict=True):
            action = cloud.actions[ref["action_id"]]
            step["uses"] = action["uses"]

    async def update_delivery(flow_id, delivery, key, **kwargs):
        assert not cloud.flow["enabled"]
        assert delivery is None or any(
            step["uses"] == "files.upload" for step in cloud.document["steps"]
        )
        cloud.document["settings"]["delivery"] = delivery
        cloud.hit("delivery")

    cloud.replace_flow_actions = replace_typed
    cloud.update_flow_delivery = AsyncMock(side_effect=update_delivery)
    cloud.prepare_action_payloads.side_effect = lambda document, **kwargs: [
        {
            "name": step["name"],
            "uses": step["uses"],
            "params": step["with"],
        }
        for step in document["steps"]
    ]
    for folder in (None, "first", "second", None):
        artifact = replace(
            changed.artifact,
            output_schema={
                "type": "object",
                "properties": {"image": {"type": "string"}},
                "required": ["image"],
            },
            declaration=CloudDeclaration(
                "demo",
                webhook=HOOK_ID,
                files_folder=folder,
                files={"map.jpg": "image"} if folder else None,
            ),
        )
        if folder == "first" or (
            folder is None and cloud.document["settings"].get("delivery")
        ):
            cloud.fail = "delivery"
            with pytest.raises(SandboxDeploymentError):
                await deploy.deploy_file(Path("fake.py"), artifact=artifact)
            assert not cloud.flow["enabled"]
        result = await deploy.deploy_file(Path("fake.py"), artifact=artifact)
        expected = (
            {"destination": "mobilerun", "folder": f"/{folder}"} if folder else None
        )
        assert cloud.document["settings"].get("delivery") == expected
        assert [step["uses"] for step in cloud.document["steps"]] == (
            ["functions.run", "files.upload", "webhooks.send"]
            if folder
            else ["functions.run", "webhooks.send"]
        )
        assert cloud.document["steps"][-1]["with"]["data"] == {
            "result": (
                {
                    "image": {
                        "path": f"/{folder}/map.jpg",
                        "fileId": "{{steps.upload_1.body.fileId}}",
                    }
                }
                if folder
                else "{{steps.function.body.result}}"
            )
        }
        assert result.action_ids == tuple(cloud.bindings)
        assert result.version == 1 and result.flow_id == "flow"
        changed.store.yaml_path.unlink()
        await sandbox_flows.set_enabled("demo", True)
        assert (
            await deploy.deploy_file(Path("fake.py"), artifact=artifact)
        ).status == "unchanged"
    added, add_retry, _, removed, remove_retry = (
        cloud.update_flow_delivery.await_args_list
    )
    assert added == add_retry and removed == remove_retry


@pytest.mark.anyio
@pytest.mark.parametrize("authored_yaml", [True, False])
async def test_redeploy_upgrades_legacy_file_webhook_order(
    changed, hooks, authored_yaml
):
    cloud, store = changed.cloud, changed.store
    artifact = replace(
        changed.artifact,
        declaration=CloudDeclaration(
            "demo", webhook=HOOK_ID, files={"map.jpg": "image"}, files_folder="atc"
        ),
        output_schema={
            "type": "object",
            "properties": {"image": {"type": "string"}},
            "required": ["image"],
        },
    )
    document = sandbox_documents.build_document(artifact, "f", 1, device=None)
    by_id = {step["id"]: step for step in document["steps"]}
    by_id["webhook"]["with"]["data"] = {"result": "{{steps.function.body.result}}"}
    document["steps"] = [by_id[key] for key in ("function", "webhook", "upload_1")]
    cloud.document = deepcopy(document)
    cloud.flow["enabled"] = False
    cloud.bindings = ["old-function", "old-webhook", "old-upload"]
    cloud.actions = {
        action_id: {"id": action_id, "name": step["name"], "params": step["with"]}
        for action_id, step in zip(cloud.bindings, document["steps"], strict=True)
    }
    receipt = store.read()
    store.write(
        replace(
            receipt,
            action_ids=tuple(cloud.bindings),
            stages=(
                json.dumps(
                    {
                        "stage": "initial",
                        "complete": True,
                        "document": document,
                    }
                ),
            ),
        )
    )
    if authored_yaml:
        store.write_yaml(document)
    else:
        store.yaml_path.unlink()

    prepare = cloud.prepare_action_payloads
    prepare.side_effect = lambda doc, **kwargs: [
        {"name": step["name"], "params": step["with"], "uses": step["uses"]}
        for step in doc["steps"]
    ]
    replace_actions = cloud.replace_flow_actions

    async def replace_typed(flow_id, refs, key, **kwargs):
        await replace_actions(flow_id, refs, key, **kwargs)
        for step, ref in zip(cloud.document["steps"], refs, strict=True):
            step["uses"] = cloud.actions[ref["action_id"]]["uses"]

    cloud.replace_flow_actions = replace_typed

    with pytest.raises(SandboxDeploymentConflict, match="precedes file uploads"):
        await sandbox_flows.set_enabled("demo", True)
    assert not cloud.flow["enabled"]
    result = await deploy.deploy_file(Path("fake.py"), artifact=artifact)
    assert result.status == "pushed"
    assert [step["id"] for step in cloud.document["steps"]] == [
        "function",
        "upload_1",
        "webhook",
    ]
    assert cloud.document["steps"][-1]["with"]["data"] == {
        "result": {
            "image": {
                "path": "/atc/map.jpg",
                "fileId": "{{steps.upload_1.body.fileId}}",
            }
        }
    }
    assert (await sandbox_flows.set_enabled("demo", True))["enabled"] is True


@pytest.mark.anyio
@pytest.mark.parametrize("evidence", ["matching", "changed", "absent", "incomplete"])
async def test_enable_webhook_without_yaml_requires_authored_evidence(
    changed, hooks, webhook_cutover, evidence
):
    cloud, store = changed.cloud, changed.store
    await webhook_cutover(HOOK_ID)
    await sandbox_flows.set_enabled("demo", False)
    store.yaml_path.unlink()
    if evidence == "changed":
        action_id = cloud.bindings[-1]
        cloud.actions[action_id]["params"][
            "endpointId"
        ] = "abcdef00-0000-4000-8000-000000000002"
        assert cloud.document["steps"][-1]["with"]["endpointId"] != HOOK_ID
    elif evidence in ("absent", "incomplete"):
        receipt = store.read()
        stages = (
            (json.dumps({"stage": "cutover", "complete": True, "desired": {}}),)
            if evidence == "incomplete"
            else ()
        )
        store.write(replace(receipt, stages=stages))

    if evidence == "matching":
        assert (await sandbox_flows.set_enabled("demo", True))["enabled"] is True
    else:
        receipt = store.read()
        hooks.get.reset_mock()
        with pytest.raises(SandboxDeploymentConflict, match="redeploy"):
            await sandbox_flows.set_enabled("demo", True)
        assert cloud.flow["enabled"] is False
        assert store.read() == receipt
        hooks.get.assert_not_awaited()


@pytest.mark.anyio
async def test_changed_cutover(changed):
    e = changed
    result = await deploy.deploy_file(Path("fake.py"), device="phone")
    assert result.version == 2
    assert e.cloud.flow["enabled"] is True
    assert e.cloud.calls.index("disable") < e.cloud.calls.index("create")
    assert e.cloud.calls.index("enable") < e.cloud.calls.index("delete_action")
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
        await deploy.deploy_file(Path("fake.py"), device="phone")
    assert error.value.receipt.function_id == "f"
    assert e.store.read().stages
    if stage == "delete_action":
        assert e.cloud.flow["enabled"] is True
    elif stage != "enable":
        assert "enable" not in e.cloud.calls
        assert e.cloud.flow["enabled"] is False
    keys = dict(e.cloud.keys)
    await deploy.deploy_file(Path("fake.py"), device="phone")
    if keys:
        assert e.cloud.keys == keys
    assert e.cloud.flow["enabled"] is True


@pytest.mark.anyio
async def test_cleanup_retains_function_and_removes_yaml(changed):
    e = changed
    e.store.write_yaml(e.cloud.document)
    await sandbox_cleanup.cleanup_owned(e.receipt, billing="no_commerce")
    assert not e.store.yaml_path.exists()
    assert "delete_function" not in e.cloud.calls
    assert e.store.read().function_id == "f"


@pytest.mark.anyio
async def test_cleanup_continues_and_retains_failed_evidence(changed):
    e = changed
    e.cloud.fail = "delete_action"
    with pytest.raises(SandboxDeploymentError) as error:
        await sandbox_cleanup.cleanup_owned(e.receipt)
    assert error.value.receipt.action_ids == ("action",)
    assert "delete_trigger" in e.cloud.calls


@pytest.mark.anyio
async def test_uncertain_shutdown_never_creates_or_enables(changed, monkeypatch):
    e = changed

    async def no_shutdown(*args, **kwargs):
        return {"enabled": False}  # A successful response is not fresh proof.

    monkeypatch.setattr(e.cloud, "disable_flow", no_shutdown)
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.deploy_file(Path("fake.py"))
    assert "enabled=True" in str(error.value)
    assert "create" not in e.cloud.calls and "enable" not in e.cloud.calls


@pytest.mark.anyio
async def test_divergent_remote_document_fails_before_mutation(changed):
    e = changed
    e.cloud.document["on"] = {"custom": {"inputs": {"foreign": True}}}
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(Path("fake.py"))
    assert "disable" not in e.cloud.calls
    assert e.cloud.version == 1


@pytest.mark.anyio
async def test_pending_cutover_rejects_divergence(changed):
    e = changed
    e.cloud.fail = "replace"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(Path("fake.py"))
    e.cloud.document["on"] = {"custom": {"foreign": True}}
    e.cloud.calls.clear()
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(Path("fake.py"))
    assert "enable" not in e.cloud.calls and "disable" not in e.cloud.calls


@pytest.mark.anyio
async def test_deployers_conflict_on_identity_lock(changed):
    e = changed
    with e.store.lock():
        with pytest.raises(SandboxDeploymentConflict):
            await deploy.deploy_file(Path("fake.py"))
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
        await deploy.deploy_file(Path("fake.py"))
    assert error.value.receipt.action_ids == ("action", "new0")
    assert e.cloud.bindings == ["new0"]
    assert e.cloud.flow["enabled"] is False
    assert "enable" not in e.cloud.calls
    await deploy.deploy_file(Path("fake.py"))
    assert e.cloud.flow["enabled"] is True


@pytest.mark.anyio
async def test_changed_disabled_flow_stays_disabled(changed):
    e = changed
    e.cloud.flow["enabled"] = False
    result = await deploy.deploy_file(Path("fake.py"))
    assert e.cloud.flow["enabled"] is False
    assert "enable" not in e.cloud.calls
    assert result.warnings


@pytest.mark.anyio
async def test_cleanup_foreign_receipt_is_read_only(changed):
    e = changed
    with pytest.raises(SandboxDeploymentConflict):
        await sandbox_cleanup.cleanup_owned(replace(e.receipt, action_ids=("foreign",)))
    assert e.cloud.calls == []


@pytest.mark.anyio
async def test_cleanup_failed_flow_deletion_retains_yaml(changed):
    e = changed
    e.cloud.fail = "delete_flow"
    with pytest.raises(SandboxDeploymentError) as error:
        await sandbox_cleanup.cleanup_owned(e.receipt)
    assert error.value.receipt.flow_id == "flow"
    assert e.store.yaml_path.exists()


@pytest.mark.anyio
async def test_cleanup_function_explicit_opt_in(changed):
    e = changed
    await sandbox_cleanup.cleanup_owned(e.receipt, delete_function=True)
    assert "delete_function" in e.cloud.calls
    assert e.store.read().function_id is None


@pytest.mark.anyio
async def test_action_intent_is_durable_before_lost_response(changed, monkeypatch):
    e = changed
    disable = e.cloud.disable_flow
    original = e.cloud.create_action
    keys = []

    async def disable_with_journal(flow_id, **kwargs):
        pending = sandbox_receipts.pending_cutover(e.store.read())
        assert pending["action_payloads"][0]["catalog_entry_id"] == "catalog"
        return await disable(flow_id, **kwargs)

    async def create(params, key, **kwargs):
        pending = sandbox_receipts.pending_cutover(e.store.read())
        assert pending["pending"] == "create:0"
        assert pending["action_payloads"] == [params]
        assert any(key in stage for stage in e.store.read().stages)
        keys.append(key)
        return await original(params, key, **kwargs)

    monkeypatch.setattr(e.cloud, "create_action", create)
    monkeypatch.setattr(e.cloud, "disable_flow", disable_with_journal)
    e.cloud.fail = "create"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(Path("fake.py"))
    await deploy.deploy_file(Path("fake.py"))
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
    await deploy.deploy_file(Path("fake.py"), device_free=True)
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
        await deploy.deploy_file(Path("fake.py"))
    with pytest.raises(SandboxDeploymentError):
        await sandbox_cleanup.cleanup_owned(e.receipt)
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
        await deploy.deploy_file(Path("fake.py"))
    e.cloud.fail = "disable"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(Path("fake.py"))
    await deploy.deploy_file(Path("fake.py"))
    assert e.cloud.flow["enabled"] is True


@pytest.mark.anyio
async def test_old_cutover_receipt_requires_manual_recovery(changed):
    e = changed
    e.cloud.fail = "create"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(Path("fake.py"))
    pending = sandbox_receipts.pending_cutover(e.store.read())
    pending.pop("action_payloads")
    receipt = sandbox_receipts.checkpoint(e.store, e.store.read(), pending)
    yaml = e.store.yaml_path.read_bytes()
    e.cloud.calls.clear()

    with pytest.raises(SandboxDeploymentConflict) as resume:
        await deploy.deploy_file(Path("fake.py"))
    with pytest.raises(SandboxDeploymentConflict) as cleanup:
        await sandbox_cleanup.cleanup_owned(receipt, delete_function=True)
    assert str(resume.value) == str(cleanup.value)
    assert "receipt predates this version" in str(resume.value)
    assert "finish or clean up the deployment manually" in str(resume.value)
    assert "Local store and ownership" in str(resume.value)
    assert e.store.read() == receipt and e.store.yaml_path.read_bytes() == yaml
    assert not e.cloud.calls


@pytest.mark.anyio
async def test_cleanup_pending_unknown_action_reports_uncertainty(changed):
    e = changed
    e.cloud.fail = "create"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(Path("fake.py"))
    receipt, yaml, calls = (
        e.store.read(),
        e.store.yaml_path.read_bytes(),
        list(e.cloud.calls),
    )
    with pytest.raises(SandboxDeploymentError, match="retry the original deploy"):
        await sandbox_cleanup.cleanup_owned(receipt, delete_function=True)
    assert e.store.read() == receipt and e.store.yaml_path.read_bytes() == yaml
    assert e.cloud.calls == calls
    await deploy.deploy_file(Path("fake.py"))
    await sandbox_cleanup.cleanup_owned(e.store.read(), delete_function=True)
    assert not e.cloud.actions and e.store.read().function_id is None


@pytest.mark.anyio
async def test_cleanup_lost_flow_delete_resolves_on_retry(changed):
    e = changed
    e.cloud.fail = "delete_flow"
    with pytest.raises(SandboxDeploymentError):
        await sandbox_cleanup.cleanup_owned(e.receipt)
    assert e.store.yaml_path.exists()
    await sandbox_cleanup.cleanup_owned(e.store.read())
    assert not e.store.yaml_path.exists()
    assert e.store.read().flow_id is None


@pytest.mark.anyio
async def test_uncertain_enable_attempts_shutdown_and_reports_actual_state(changed):
    e = changed
    e.cloud.fail = "enable"
    with pytest.raises(SandboxDeploymentError) as error:
        await deploy.deploy_file(Path("fake.py"))
    assert e.cloud.flow["enabled"] is False
    assert "enabled=False" in str(error.value)
    assert sandbox_receipts.pending_cutover(error.value.receipt)["warning"] == str(
        error.value
    )
    assert e.cloud.calls[-1] == "disable"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "stage", ["delete_action", "delete_trigger", "delete_function"]
)
async def test_cleanup_lost_delete_response_real_404_completes_on_retry(changed, stage):
    e = changed
    e.cloud.fail = stage
    with pytest.raises(SandboxDeploymentError):
        await sandbox_cleanup.cleanup_owned(e.receipt, delete_function=True)
    await sandbox_cleanup.cleanup_owned(e.store.read(), delete_function=True)
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
