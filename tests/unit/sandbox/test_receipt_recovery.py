"""Sandbox receipt storage, recovery of deployment intent, and owned-resource cleanup."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import date
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_core import PydanticCustomError

from mobilerun.models.billing import Billing
from mobilerun.sandbox import Remote
from mobilerun.sandbox import cleanup as sandbox_cleanup
from mobilerun.sandbox import deployment as deploy
from mobilerun.sandbox import documents as sandbox_documents
from mobilerun.sandbox import flows as sandbox_flows
from mobilerun.sandbox import receipts as sandbox_receipts
from mobilerun.sandbox.errors import (
    SandboxDeclarationError,
    SandboxDeploymentConflict,
    SandboxDeploymentError,
)
from mobilerun.sandbox.models import Artifact, CloudDeclaration, OwnershipReceipt
from mobilerun.sandbox.settings import SandboxSettings
from tests.support.paths import REPO_ROOT
from tests.support.sandbox_fixtures import (
    Input,
    Output,
    backend,
    patch_sandbox,
    setup_deploy,
)
from tests.unit.sandbox.staging_fixtures import (
    InitialSandboxCloud,
    base_changed,
    base_env,
    changed,
    env,
    initial_cloud,
)

__all__ = [
    "base_changed",
    "base_env",
    "env",
    "setup_deploy",
    "changed",
    "initial_cloud",
    "backend",
]


def scope(origin="https://example.test/v1", key="secret"):
    return sandbox_receipts.Scope(origin, hashlib.sha256(key.encode()).hexdigest())


def receipt(s, name="../unsafe"):
    return OwnershipReceipt(
        s.origin, s.account_id, name, None, None, None, (), None, ()
    )


def store(tmp_path, s=None, name="../unsafe"):
    return sandbox_receipts.ReceiptStore(
        s or scope(), name, SandboxSettings(MOBILERUN_SANDBOX_DIRECTORY=str(tmp_path))
    )


def test_scope_uses_cloud_client_key_resolver_and_no_secret(monkeypatch):
    monkeypatch.setattr(sandbox_receipts, "resolve_cloud_api_key", lambda: "sensitive")
    monkeypatch.setattr(
        sandbox_receipts,
        "get_settings",
        lambda: SimpleNamespace(MOBILERUN_CLOUD_BASE_URL="https://a"),
    )
    s = sandbox_receipts.current_scope()
    assert s.origin == "https://a/"
    assert s.account_id == hashlib.sha256(b"sensitive").hexdigest()
    assert "sensitive" not in repr(s)
    monkeypatch.setattr(sandbox_receipts, "resolve_cloud_api_key", lambda: None)
    with pytest.raises(SandboxDeploymentError, match="credentials"):
        sandbox_receipts.current_scope()


@pytest.mark.parametrize("override", [None, "", "relative/store", "~/cloud-store"])
def test_path_settings_default_and_overrides(tmp_path, monkeypatch, override):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "mobilerun.sandbox.settings.get_settings",
        lambda: SimpleNamespace(MOBILERUN_USER_CONFIG_DIR=tmp_path / "config"),
    )
    settings = SandboxSettings(MOBILERUN_SANDBOX_DIRECTORY=override)
    st = sandbox_receipts.ReceiptStore(scope(), "../unsafe", settings)
    root = {
        None: tmp_path / "config/sandbox",
        "": tmp_path / "config/sandbox",
        "relative/store": tmp_path / "relative/store",
        "~/cloud-store": tmp_path / "cloud-store",
    }[override]
    assert (
        st.directory
        == root
        / hashlib.sha256(
            (scope().origin + "\0" + scope().account_id).encode()
        ).hexdigest()
        / hashlib.sha256(b"../unsafe").hexdigest()
    )
    assert st.read() is None
    assert not root.exists()


def test_scope_origin_account_and_names_are_isolated(tmp_path):
    assert (
        len(
            {
                store(tmp_path, scope(), "n").directory,
                store(tmp_path, scope(key="other"), "n").directory,
                store(tmp_path, scope(origin="https://other"), "n").directory,
                store(tmp_path, scope(), "../n").directory,
            }
        )
        == 4
    )


def test_atomic_write_preserves_old_receipt_on_replace_failure(tmp_path, monkeypatch):
    st = store(tmp_path)
    r = receipt(scope())
    st.write(r)

    def crash(*args):
        raise OSError("secret detail")

    monkeypatch.setattr(sandbox_receipts.os, "replace", crash)
    with pytest.raises(SandboxDeploymentError) as exc:
        st.write(replace(r, function_id="new"))
    assert "secret detail" not in str(exc.value)
    assert st.read() == r
    assert list(st.directory.glob(".sandbox-*"))


@pytest.mark.parametrize("bad", ["not json", "{}", '{"origin": 3}', "[]"])
def test_malformed_receipts_fail_closed(tmp_path, bad):
    st = store(tmp_path)
    st.directory.mkdir(parents=True)
    st.receipt_path.write_text(bad)
    with pytest.raises(SandboxDeploymentError):
        st.read()


def test_scope_and_name_mismatch_cannot_be_adopted(tmp_path):
    st = store(tmp_path)
    st.directory.mkdir(parents=True)
    st.receipt_path.write_text(json.dumps(receipt(scope(key="different")).to_dict()))
    with pytest.raises(SandboxDeploymentConflict):
        st.read()


def test_identity_lock_conflicts_and_releases(tmp_path):
    st = store(tmp_path)
    with st.lock():
        with pytest.raises(SandboxDeploymentConflict):
            with st.lock():
                pass
    with st.lock():
        pass


def test_stage_intent_and_ids_are_persisted_and_stable(tmp_path):
    s = scope()
    st = store(tmp_path)
    r = receipt(s)
    key = sandbox_receipts.stage_key(s, r.name, "digest", "action:create:0")
    assert key == sandbox_receipts.stage_key(s, r.name, "digest", "action:create:0")
    assert key != sandbox_receipts.stage_key(s, r.name, "digest", "action:create:1")
    r = st.record_stage(r, "action:create:0", key)
    assert st.read() == r
    assert key in r.stages[-1] and "action:create:0" in r.stages[-1]
    r = st.record_ids(r, function_id="f", action_ids=("a",))
    assert st.read() == r
    assert r.function_id == "f" and r.action_ids == ("a",)


def test_yaml_quotes_on_and_only_removes_with_confirmation(tmp_path):
    st = store(tmp_path)
    st.write_yaml({"on": {"custom": {}}, "version": 1})
    assert '"on":' in st.yaml_path.read_text()
    assert yaml.safe_load(st.yaml_path.read_text())["on"] == {"custom": {}}
    with pytest.raises(SandboxDeploymentError):
        st.remove_yaml(cleanup_confirmed=False)
    assert st.yaml_path.exists()
    st.remove_yaml(cleanup_confirmed=True)
    assert not st.yaml_path.exists()


def test_windows_lock_backend_is_selected_and_released(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []
    fake = SimpleNamespace(
        LK_NBLCK=1,
        LK_UNLCK=2,
        locking=lambda fd, mode, size: calls.append((mode, size)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(
        sandbox_receipts, "_windows_platform", lambda: True, raising=False
    )
    with store(tmp_path).lock():
        assert calls == [(1, 1)]
    assert calls == [(1, 1), (2, 1)]


@pytest.fixture
def _inspect_synthetic_sources(monkeypatch):
    # Recovery fixtures use strings like "different" as opaque function sources.
    monkeypatch.setattr(deploy, "verify_scheduled_artifact", lambda artifact: None)


async def interrupt_schedule(e):
    scheduled = replace(
        e.artifact, declaration=CloudDeclaration("demo", schedule="* * * * *")
    )
    e.cloud.fail = "replace"
    with pytest.raises(SandboxDeploymentError, match="Cutover incomplete"):
        await deploy.deploy_file(Path("demo.py"), artifact=scheduled, device_free=True)
    return scheduled


@pytest.mark.usefixtures("_inspect_synthetic_sources")
@pytest.mark.anyio
async def test_changed_trigger_cannot_resume_pending_schedule(changed):
    await interrupt_schedule(changed)
    before = list(changed.cloud.calls)
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(
            Path("demo.py"), artifact=changed.artifact, device_free=True
        )
    assert changed.cloud.flow["enabled"] is False
    assert set(changed.cloud.calls[len(before) :]) <= {"read_document"}


@pytest.mark.usefixtures("_inspect_synthetic_sources")
@pytest.mark.anyio
async def test_pause_survives_pending_schedule_recovery(changed):
    scheduled = await interrupt_schedule(changed)
    assert not (await sandbox_flows.set_enabled("demo", False))["enabled"]
    await deploy.deploy_file(Path("demo.py"), artifact=scheduled, device_free=True)
    assert changed.cloud.flow["enabled"] is False


@pytest.mark.usefixtures("_inspect_synthetic_sources")
@pytest.mark.anyio
@pytest.mark.parametrize("pause", [False, True])
async def test_initial_enable_recovery_respects_pause(
    initial_cloud, monkeypatch, pause
):
    e = initial_cloud
    with monkeypatch.context() as patch:
        patch.setattr(e.cloud, "enable_flow", AsyncMock(side_effect=RuntimeError()))
        with pytest.raises(SandboxDeploymentError, match="activation"):
            await deploy.deploy_file(e.path)
    if pause:
        await sandbox_flows.set_enabled("demo", False)
    await deploy.deploy_file(e.path)
    assert e.cloud.flow["enabled"] is (not pause)


@pytest.mark.usefixtures("_inspect_synthetic_sources")
@pytest.mark.anyio
async def test_enable_ack_without_remote_activation_stays_pending(
    initial_cloud, monkeypatch
):
    e = initial_cloud
    with monkeypatch.context() as patch:
        patch.setattr(e.cloud, "enable_flow", AsyncMock(return_value={"enabled": True}))
        with pytest.raises(SandboxDeploymentError, match="activation"):
            await deploy.deploy_file(e.path)
    await deploy.deploy_file(e.path)
    assert e.cloud.flow["enabled"]


@pytest.mark.usefixtures("_inspect_synthetic_sources")
@pytest.mark.anyio
async def test_full_cleanup_closes_interrupted_replacement(initial_cloud):
    e = initial_cloud
    await deploy.deploy_file(e.path)
    await interrupt_schedule(e)
    await sandbox_cleanup.cleanup_owned(e.store.read(), delete_function=True)
    await deploy.deploy_file(e.path)
    assert (await sandbox_flows.set_enabled("demo", True))["enabled"]


@pytest.mark.usefixtures("_inspect_synthetic_sources")
@pytest.mark.anyio
async def test_corrected_source_can_replace_rejected_candidate(changed, monkeypatch):
    e = changed

    async def inspect(function_id, version, **kw):
        return {"version": version, "status": "rejected" if version == 2 else "ready"}

    async def upload(function_id, source, key, **kw):
        assert e.cloud.flow["enabled"]
        assert e.cloud.document["steps"][0]["with"]["version"] == 1
        e.cloud.version += 1
        e.cloud.source = source
        return {"version": e.cloud.version}

    monkeypatch.setattr(e.cloud, "get_version", inspect)
    monkeypatch.setattr(e.cloud, "create_version", upload)
    with pytest.raises(SandboxDeploymentError, match="rejected"):
        await deploy.deploy_file(Path("demo.py"))
    corrected = replace(e.artifact, source="corrected", digest="corrected")
    result = await deploy.deploy_file(Path("demo.py"), artifact=corrected)
    assert result.version == 3 and e.cloud.flow["enabled"]


@pytest.mark.usefixtures("_inspect_synthetic_sources")
@pytest.mark.parametrize("direction", ["input", "output"])
def test_typed_validation_hides_raw_payload(backend, direction):
    handle = Remote("x", Input, Output)
    marker = "synthetic-private-value"
    with pytest.raises(ValidationError) as caught:
        if direction == "input":
            handle.fire(n=marker)
        else:
            backend["wait_result"].return_value["result"]["steps"][0]["body"][
                "result"
            ] = {"n": marker}
            handle.fire(n=2).result()
    assert marker not in str(caught.value) + caught.value.json()
    assert caught.value.__context__ is None
    assert [(e["loc"], e["type"]) for e in caught.value.errors()] == [
        (("n",), "int_parsing")
    ]


@pytest.mark.usefixtures("_inspect_synthetic_sources")
def test_dynamic_locations_do_not_expose_payload(backend):
    class A(BaseModel):
        kind: Literal["synthetic-tag"]
        count: int

    class B(BaseModel):
        kind: Literal["other"]

    class Payload(BaseModel):
        model_config = ConfigDict(extra="forbid")
        alias_only: int = Field(alias="declared_alias", default=1)
        fields: dict[str, int]
        numbers: list[int]
        choice: A | B = Field(discriminator="kind")

    with pytest.raises(ValidationError) as caught:
        Remote("demo", Payload).fire(
            input={
                "fields": {"synthetic-key": "bad"},
                "numbers": ["bad"],
                "choice": {"kind": "synthetic-tag", "count": "bad"},
                "synthetic-extra": 1,
                "alias_only": 1,
            }
        )
    assert "synthetic-" not in caught.value.json()
    assert "alias_only" not in caught.value.json()
    locations = [error["loc"] for error in caught.value.errors()]
    assert ("fields", "<key>") in locations
    assert ("numbers", 0) in locations
    assert ("<key>",) in locations


@pytest.mark.usefixtures("_inspect_synthetic_sources")
@pytest.mark.anyio
async def test_replace_terminally_rejected_first_deploy(initial_cloud, monkeypatch):
    e = initial_cloud

    async def inspect(function_id, version, **kw):
        return {"version": version, "status": "rejected" if version == 1 else "ready"}

    monkeypatch.setattr(e.cloud, "get_version", inspect)
    with pytest.raises(SandboxDeploymentError, match="rejected"):
        await deploy.deploy_file(e.path)
    assert not e.cloud.flow
    result = await deploy.deploy_file(
        e.path,
        artifact=replace(e.artifact, source="corrected", digest="corrected"),
    )
    assert result.function_id == "f" and result.version == 2
    assert e.cloud.flow["enabled"] and e.cloud.source == "corrected"
    assert sum(stage == "function" for stage, _ in e.cloud.create_calls) == 1


def remove_preview(store):
    receipt = store.read()
    legacy = [json.loads(entry) for entry in receipt.stages]
    for entry in legacy:
        entry.pop("preview_document", None)
    store.write(replace(receipt, stages=tuple(json.dumps(entry) for entry in legacy)))


@pytest.mark.usefixtures("_inspect_synthetic_sources")
@pytest.mark.anyio
@pytest.mark.parametrize("foreign_flow", [False, True])
async def test_legacy_initial_recovers_only_function_for_cleanup(
    initial_cloud, foreign_flow
):
    e = initial_cloud
    e.cloud.fail = "function"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(e.path)
    remove_preview(e.store)
    expected_flow = None
    if foreign_flow:
        expected_flow = {
            "id": "unrelated",
            "name": e.cloud.document["name"],
            "triggerId": "theirs",
            "enabled": True,
        }
        e.cloud.flow = expected_flow.copy()
        e.cloud.document["steps"][0]["with"]["functionId"] = "theirs"
    before = list(e.cloud.create_calls)
    schedule = replace(
        e.artifact, declaration=CloudDeclaration("demo", schedule="* * * * *")
    )
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(
            e.path,
            artifact=replace(schedule, digest="other", source="different"),
            device_free=True,
        )
    assert e.cloud.create_calls == before
    with pytest.raises(SandboxDeploymentConflict, match="cleanup") as caught:
        await deploy.deploy_file(e.path, artifact=schedule, device_free=True)
    assert e.cloud.create_calls == before * 2
    assert (
        caught.value.receipt == e.store.read()
        and caught.value.receipt.function_id == "f"
    )
    await sandbox_cleanup.cleanup_owned(e.store.read(), delete_function=True)
    assert not e.cloud.function_exists and e.cloud.flow == expected_flow
    assert set(e.cloud.calls) <= {"read_document", "delete_function"}


@pytest.mark.usefixtures("_inspect_synthetic_sources")
@pytest.mark.anyio
async def test_legacy_initial_uses_pinned_document(initial_cloud):
    e = initial_cloud
    e.cloud.fail = "flow"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(e.path)
    remove_preview(e.store)
    before = list(e.cloud.create_calls)
    schedule = replace(
        e.artifact, declaration=CloudDeclaration("demo", schedule="* * * * *")
    )
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(e.path, artifact=schedule, device_free=True)
    assert e.cloud.create_calls == before
    assert (await deploy.deploy_file(e.path)).endpoint


@pytest.mark.usefixtures("_inspect_synthetic_sources")
@pytest.mark.anyio
@pytest.mark.parametrize("status", ["ready", "pending"])
async def test_first_deploy_supersession_rechecks_rejection(
    initial_cloud, monkeypatch, status
):
    e = initial_cloud
    inspect = AsyncMock(return_value={"version": 1, "status": "rejected"})
    monkeypatch.setattr(e.cloud, "get_version", inspect)
    with pytest.raises(SandboxDeploymentError, match="rejected"):
        await deploy.deploy_file(e.path)
    inspect.return_value = {"version": 1, "status": status}
    before = e.store.read()
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(
            e.path,
            artifact=replace(e.artifact, source="corrected", digest="corrected"),
        )
    assert e.store.read() == before and e.cloud.version == 1 and not e.cloud.flow


@pytest.mark.usefixtures("_inspect_synthetic_sources")
def test_custom_error_type_cannot_disclose_input(backend):
    class Payload(BaseModel):
        token: str

        @field_validator("token")
        @classmethod
        def reject(cls, value: str) -> str:
            raise PydanticCustomError(value, "rejected")

    with pytest.raises(ValidationError) as caught:
        Remote("demo", Payload).fire(token="synthetic-private-code")
    assert "synthetic-private-code" not in str(caught.value) + caught.value.json()
    assert caught.value.errors()[0]["type"] == "invalid"


@pytest.mark.usefixtures("_inspect_synthetic_sources")
@pytest.mark.anyio
async def test_initial_schema_const_requires_same_json_type(initial_cloud):
    e = initial_cloud
    e.cloud.fail = "function"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(
            e.path, artifact=replace(e.artifact, input_schema={"const": True})
        )
    before = list(e.cloud.create_calls)
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(
            e.path, artifact=replace(e.artifact, input_schema={"const": 1})
        )
    assert e.cloud.create_calls == before


@pytest.mark.usefixtures("_inspect_synthetic_sources")
@pytest.mark.anyio
async def test_cutover_schema_const_requires_same_json_type(changed):
    e = changed
    e.cloud.fail = "replace"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(
            Path("demo.py"),
            artifact=replace(e.artifact, input_schema={"const": True}),
        )
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(
            Path("demo.py"),
            artifact=replace(e.artifact, input_schema={"const": 1}),
        )
    assert not e.cloud.flow["enabled"]


@pytest.mark.usefixtures("_inspect_synthetic_sources")
def test_remote_document_bool_is_not_number():
    assert not sandbox_documents.equivalent_document(
        {"on": {"custom": {"inputs": {"const": True}}}},
        {"on": {"custom": {"inputs": {"const": 1}}}},
    )


@pytest.mark.usefixtures("_inspect_synthetic_sources")
@pytest.mark.anyio
async def test_initial_receipt_accepts_equivalent_number(initial_cloud):
    e = initial_cloud
    e.cloud.fail = "function"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(
            e.path,
            artifact=replace(e.artifact, input_schema={"minimum": 0.0}),
        )
    result = await deploy.deploy_file(
        e.path, artifact=replace(e.artifact, input_schema={"minimum": 0})
    )
    assert result.flow_id and e.cloud.flow["enabled"]


@pytest.mark.usefixtures("_inspect_synthetic_sources")
def test_remote_document_accepts_equivalent_number():
    assert sandbox_documents.equivalent_document(
        {"on": {"custom": {"inputs": {"minimum": 0}}}, "version": 1},
        {"on": {"custom": {"inputs": {"minimum": 0.0}}}, "version": 1.0},
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    deployment = import_module("mobilerun.sandbox.deployment")
    name = "demo"
    source = tmp_path / "ping.py"
    artifact = Artifact(
        CloudDeclaration(name, endpoint=True),
        "source",
        "digest",
        {"type": "object"},
        None,
    )
    scope = sandbox_receipts.Scope("https://example.test/v1", "a" * 64)
    monkeypatch.setenv("MOBILERUN_SANDBOX_DIRECTORY", str(tmp_path / "sandbox"))
    patch_sandbox(monkeypatch, "current_scope", lambda: scope)
    monkeypatch.setattr(deployment, "package", lambda path: artifact)
    document = sandbox_documents.build_document(artifact, "f", 1, device=None)
    cloud = InitialSandboxCloud(document)
    patch_sandbox(monkeypatch, "clients", cloud)
    return deployment, source, sandbox_receipts.ReceiptStore(scope, name), cloud


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("failed_at", "ids"),
    [
        ("function", (None, None, None, ())),
        ("trigger", ("f", None, None, ())),
        ("action", ("f", None, "trigger", ())),
        ("flow", ("f", None, "trigger", ("action",))),
    ],
)
async def test_cleanup_fails_closed_after_lost_creation_response(setup, failed_at, ids):
    deployment, source, store, cloud = setup
    cloud.fail = failed_at
    with pytest.raises(SandboxDeploymentError):
        await deployment.deploy_file(source, billing=Billing.PAID)

    receipt = store.read()
    assert isinstance(receipt, OwnershipReceipt)
    assert (
        receipt.function_id,
        receipt.flow_id,
        receipt.trigger_id,
        receipt.action_ids,
    ) == ids
    before = list(cloud.calls)
    with pytest.raises(SandboxDeploymentError, match="creation outcome uncertain"):
        await sandbox_cleanup.cleanup_owned(
            receipt, delete_function=True, billing=Billing.NO_COMMERCE
        )
    assert cloud.calls == before
    assert store.read() == receipt

    # Only in the fake: replay the original intent to recover the missing IDs.
    await deployment.deploy_file(source, billing=Billing.PAID)
    await sandbox_cleanup.cleanup_owned(
        store.read(), delete_function=True, billing=Billing.NO_COMMERCE
    )
    assert cloud.flow is None and not cloud.trigger_exists
    assert not cloud.function_exists and not cloud.actions


@pytest.mark.anyio
@pytest.mark.parametrize("failed_at", ["validation", "enable"])
async def test_cleanup_removes_recorded_resources_after_known_creation(
    setup, failed_at, monkeypatch
):
    deployment, source, store, cloud = setup
    if failed_at == "validation":

        async def invalid(document, *, billing):
            return {"valid": False}

        monkeypatch.setattr(cloud, "validate_document", invalid)
    else:
        cloud.fail = "enable"
    with pytest.raises(SandboxDeploymentError):
        await deployment.deploy_file(source, billing=Billing.PAID)
    receipt = store.read()
    assert isinstance(receipt, OwnershipReceipt)
    assert receipt.function_id == "f"
    assert receipt.flow_id == ("flow" if failed_at == "enable" else None)

    await sandbox_cleanup.cleanup_owned(
        receipt, delete_function=True, billing=Billing.NO_COMMERCE
    )
    expected = ["delete_function"]
    if failed_at == "enable":
        expected = [
            "disable",
            "delete_flow",
            "delete_action",
            "delete_trigger",
            *expected,
        ]
    assert cloud.calls[-len(expected) :] == expected
    assert cloud.flow is None and not cloud.trigger_exists
    assert not cloud.function_exists and not cloud.actions
    assert not store.yaml_path.exists()


@pytest.mark.anyio
async def test_same_name_preexisting_function_never_becomes_cleanup_target(setup):
    deployment, source, store, cloud = setup
    cloud.function_exists = True
    with pytest.raises(SandboxDeploymentConflict, match="not owned"):
        await deployment.deploy_file(source, billing=Billing.PAID)
    assert store.read() is None
    assert cloud.function_exists and not cloud.calls and not cloud.create_calls


@pytest.mark.anyio
async def test_unattached_foreign_action_retains_owned_function(setup):
    deployment, source, store, cloud = setup
    await deployment.deploy_file(source)
    cloud.actions["foreign"] = {
        "id": "foreign",
        "uses": "functions.run",
        "params": {"functionId": "f", "version": 1},
    }
    with pytest.raises(SandboxDeploymentError) as error:
        await sandbox_cleanup.cleanup_owned(
            store.read(), delete_function=True, billing=Billing.NO_COMMERCE
        )
    assert error.value.receipt.function_id == "f"
    assert "f" in str(error.value)
    assert cloud.function_exists and "delete_function" not in cloud.calls
    assert "foreign" in cloud.actions and cloud.flow is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "definition",
    [
        None,
        {
            "id": "foreign",
            "uses": "functions.run",
            "params": {"functionId": "{{event.payload}}"},
        },
        {"id": "foreign", "uses": None, "params": None},
    ],
)
async def test_unknown_action_sharing_fails_closed(setup, monkeypatch, definition):
    deployment, source, store, cloud = setup
    await deployment.deploy_file(source)

    async def definitions(**kwargs):
        if definition is None:
            raise RuntimeError("action listing unavailable")
        return [definition]

    monkeypatch.setattr(cloud, "list_action_definitions", definitions, raising=False)
    with pytest.raises(SandboxDeploymentError):
        await sandbox_cleanup.cleanup_owned(store.read(), delete_function=True)
    assert cloud.function_exists and "delete_function" not in cloud.calls


@pytest.mark.anyio
@pytest.mark.parametrize(
    "definition",
    [
        {"id": "foreign", "uses": "functions.run", "params": {"functionId": "other"}},
        {"id": "foreign", "uses": "devices.list", "params": None},
    ],
)
async def test_unrelated_unattached_action_allows_owned_function_cleanup(
    setup, definition
):
    deployment, source, store, cloud = setup
    await deployment.deploy_file(source)
    cloud.actions["foreign"] = definition
    await sandbox_cleanup.cleanup_owned(store.read(), delete_function=True)
    assert not cloud.function_exists
    assert cloud.actions == {"foreign": definition}


@pytest.mark.anyio
async def test_action_definition_reader_checks_all_pages_without_commerce():
    from mobilerun.sandbox import clients

    listing = AsyncMock(
        side_effect=[
            {"items": [{"id": "one"}], "pagination": {"hasNext": True}},
            {"items": [{"id": "two"}], "pagination": {"hasNext": False}},
        ]
    )
    sdk = SimpleNamespace(
        workflows=SimpleNamespace(actions=SimpleNamespace(list=listing))
    )
    rows = await clients.list_action_definitions(
        billing=Billing.NO_COMMERCE, client=sdk
    )
    assert [row["id"] for row in rows] == ["one", "two"]
    assert [call.kwargs["page"] for call in listing.await_args_list] == [1, 2]
    assert all(
        set(call.kwargs) == {"page", "page_size"} for call in listing.await_args_list
    )


def test_smoke_origin_survives_collection_isolation():
    code = """
import os, runpy, pytest
expected = "https://staging.example.test/v1"
os.environ["MOBILERUN_CLOUD_BASE_URL"] = expected
scope = runpy.run_path("tests/conftest.py")
assert "MOBILERUN_CLOUD_BASE_URL" not in os.environ
with pytest.MonkeyPatch.context() as patch:
    scope["_real_secrets"].apply(patch)
    assert os.environ.get("MOBILERUN_CLOUD_BASE_URL") == expected
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "origin",
    [
        None,
        "",
        "not-a-url",
        "https://example.test/v1?token=value",
        "https://user:password@example.test/v1",
    ],
)
def test_smoke_requires_explicit_unambiguous_origin(monkeypatch, origin):
    smoke = import_module("tests.unit.smoke.test_sandbox_functions")
    if origin is None:
        monkeypatch.delenv("MOBILERUN_CLOUD_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("MOBILERUN_CLOUD_BASE_URL", origin)
    with pytest.raises(pytest.skip.Exception, match="MOBILERUN_CLOUD_BASE_URL"):
        smoke._verified_smoke_scope()


def test_smoke_verifies_origin_before_deployment(monkeypatch):
    smoke = import_module("tests.unit.smoke.test_sandbox_functions")
    expected = "https://staging.example.test/v1"
    monkeypatch.setenv("MOBILERUN_CLOUD_BASE_URL", expected)
    scope = sandbox_receipts.Scope(expected, "fingerprint")
    patch_sandbox(monkeypatch, "current_scope", lambda: scope)
    assert smoke._verified_smoke_scope() == scope
    patch_sandbox(
        monkeypatch,
        "current_scope",
        lambda: sandbox_receipts.Scope("https://other.example.test/v1", "fingerprint"),
    )
    with pytest.raises(AssertionError, match="origin"):
        smoke._verified_smoke_scope()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "execution_id", "delete"),
    [
        ("succeeded", "execution", True),
        ("failed", "execution", True),
        ("cancelled", "execution", True),
        ("uploading", "execution", False),
        ("unknown", "execution", False),
        ("unrelated", "execution", False),
        ("unavailable", "execution", False),
        (None, "execution", False),
        (None, None, False),
    ],
)
async def test_smoke_file_cleanup_after_copy_timeout(status, execution_id, delete):
    smoke = import_module("tests.unit.smoke.test_sandbox_functions")
    wait = AsyncMock(side_effect=TimeoutError)
    retrieve = AsyncMock(return_value={"status": "success", "deliveries": []})
    if status == "unavailable":
        retrieve.side_effect = TimeoutError
    elif status is not None:
        retrieve.return_value["deliveries"].append(
            {
                "destination": "mobilerun",
                "artifact": "file",
                "folder": "/smoke",
                "filename": "other.png" if status == "unrelated" else "pixel.png",
                "status": "succeeded" if status == "unrelated" else status,
            }
        )
    clean = AsyncMock()
    kwargs = dict(wait_for_copy=wait, get_execution=retrieve, clean=clean)
    if delete:
        await smoke._cleanup_saved_copy("/smoke", execution_id, **kwargs)
        clean.assert_awaited_once()
    else:
        with pytest.warns(
            RuntimeWarning,
            match=f"unconfirmed.*folder=/smoke.*execution={execution_id or 'unknown'}",
        ):
            await smoke._cleanup_saved_copy("/smoke", execution_id, **kwargs)
        clean.assert_not_awaited()
    assert retrieve.await_count == int(execution_id is not None)


@pytest.mark.anyio
async def test_smoke_file_cleanup_after_observing_copy():
    smoke = import_module("tests.unit.smoke.test_sandbox_functions")
    retrieve = AsyncMock()
    clean = AsyncMock()
    await smoke._cleanup_saved_copy(
        "/smoke", None, wait_for_copy=AsyncMock(), get_execution=retrieve, clean=clean
    )
    retrieve.assert_not_awaited()
    clean.assert_awaited_once()


@pytest.mark.anyio
async def test_disable_without_document(changed):
    changed.cloud.fail = "read_document"
    assert not (await sandbox_flows.set_enabled("demo", False))["enabled"]
    assert changed.cloud.calls == ["disable"]


@pytest.mark.anyio
async def test_rollback_then_identical_deploy(changed, monkeypatch):
    cloud = changed.cloud
    keys, sources = {}, {1: "source"}

    async def upload(fid, source, key, **kwargs):
        if key not in keys:
            cloud.version += 1
            sources[cloud.version] = source
            keys[key] = cloud.version
        return {"version": keys[key]}

    async def get_source(fid, version, **kwargs):
        return sources[version]

    monkeypatch.setattr(cloud, "create_version", upload)
    monkeypatch.setattr(cloud, "get_source", get_source)
    for src in ["B", "C", "B", "B"]:
        cloud.calls.clear()
        result = await deploy.deploy_file(
            Path("demo.py"),
            artifact=replace(changed.artifact, source=src, digest=src),
        )
    assert result.status == "unchanged"
    assert result.version == 4 and cloud.calls == ["read_document"]
    # Another caller may have pushed an unpinned version since the last deploy.
    cloud.version += 1
    sources[cloud.version] = "other"
    cloud.calls.clear()
    result = await deploy.deploy_file(
        Path("demo.py"),
        artifact=replace(changed.artifact, source="B", digest="B"),
    )
    assert result.version == 4 and cloud.calls == ["read_document"]


def test_url_aliases_recover_ownership_or_fail_closed(tmp_path, monkeypatch):
    base = "https://example.test/v1"
    monkeypatch.setattr(sandbox_receipts, "resolve_cloud_api_key", lambda: "key")
    monkeypatch.setenv("MOBILERUN_SANDBOX_DIRECTORY", str(tmp_path))

    def scoped(url):
        monkeypatch.setattr(
            sandbox_receipts,
            "get_settings",
            lambda: SimpleNamespace(MOBILERUN_CLOUD_BASE_URL=url),
        )
        return sandbox_receipts.current_scope()

    assert scoped(base) == scoped(base + "/")
    for url, fid in [(base, "f"), (base + "/", "other")]:
        account = scoped(url).account_id
        digest = hashlib.sha256((url + "\0" + account).encode()).hexdigest()
        directory = tmp_path / digest / hashlib.sha256(b"demo").hexdigest()
        directory.mkdir(parents=True)
        receipt = OwnershipReceipt(url, account, "demo", fid, None, None, (), None, ())
        (directory / "receipt.json").write_text(json.dumps(receipt.to_dict()))
        if fid == "f":
            assert (
                sandbox_receipts.ReceiptStore(scoped(base + "/"), "demo")
                .read()
                .function_id
                == "f"
            )
    with pytest.raises(SandboxDeploymentConflict):
        sandbox_receipts.ReceiptStore(scoped(base), "demo").read()


@pytest.mark.anyio
async def test_schedule_rejects_required_input_before_mutation(changed):
    art = replace(
        changed.artifact,
        declaration=CloudDeclaration("demo", schedule="* * * * *"),
        source="from mobilerun_functions import script\nfrom pydantic import BaseModel\nclass Required(BaseModel):\n    count: int\n@script\ndef task(input: Required) -> Required: return input\n",
        input_schema={"type": "object"},  # No model-inspection evidence.
    )
    with pytest.raises(SandboxDeclarationError, match="count"):
        await deploy.deploy_file(Path("demo.py"), artifact=art, device_free=True)
    assert not changed.cloud.calls


def test_date_result_decodes_in_json_mode(backend):
    class Out(BaseModel):
        model_config = ConfigDict(strict=True)
        day: date

    backend["wait_result"].return_value["result"]["steps"][0]["body"]["result"] = {
        "day": "2026-01-01"
    }
    assert Remote("demo", Input, Out).fire(n=1).result().day == date(2026, 1, 1)
