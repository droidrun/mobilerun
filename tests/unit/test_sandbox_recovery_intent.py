"""Regression checks for the intent preserved across deployment failures."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_core import PydanticCustomError

from mobilerun.sandbox import Remote
from mobilerun.sandbox import deployment as deploy
from mobilerun.sandbox.errors import SandboxDeploymentConflict, SandboxDeploymentError
from mobilerun.sandbox.models import CloudDeclaration
from tests.unit.test_sandbox_first_deploy import env, initial_cloud
from tests.unit.test_sandbox_redeploy import changed, setup_deploy
from tests.unit.test_sandbox_remote import Input, Output, backend

__all__ = ["env", "setup_deploy", "changed", "initial_cloud", "backend"]


@pytest.fixture(autouse=True)
def _inspect_synthetic_sources(monkeypatch):
    # Recovery fixtures use strings like "different" as opaque function sources.
    monkeypatch.setattr(deploy, "verify_scheduled_artifact", lambda artifact: None)


async def interrupt_schedule(e):
    scheduled = replace(
        e.artifact, declaration=CloudDeclaration("demo", schedule="* * * * *")
    )
    e.cloud.fail = "replace"
    with pytest.raises(SandboxDeploymentError, match="Cutover incomplete"):
        await deploy.deploy_file(
            Path("demo.py"), artifact=scheduled, device_free=True, yes=True
        )
    return scheduled


@pytest.mark.anyio
async def test_changed_trigger_cannot_resume_pending_schedule(changed):
    await interrupt_schedule(changed)
    before = list(changed.cloud.calls)
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(
            Path("demo.py"), artifact=changed.artifact, device_free=True, yes=True
        )
    assert changed.cloud.flow["enabled"] is False
    assert set(changed.cloud.calls[len(before) :]) <= {"read_document"}


@pytest.mark.anyio
async def test_pause_survives_pending_schedule_recovery(changed):
    scheduled = await interrupt_schedule(changed)
    assert not (await deploy.set_enabled("demo", False))["enabled"]
    await deploy.deploy_file(
        Path("demo.py"), artifact=scheduled, device_free=True, yes=True
    )
    assert changed.cloud.flow["enabled"] is False


@pytest.mark.anyio
@pytest.mark.parametrize("pause", [False, True])
async def test_initial_enable_recovery_respects_pause(
    initial_cloud, monkeypatch, pause
):
    e = initial_cloud
    with monkeypatch.context() as patch:
        patch.setattr(e.cloud, "enable_flow", AsyncMock(side_effect=RuntimeError()))
        with pytest.raises(SandboxDeploymentError, match="activation"):
            await deploy.deploy_file(e.path, yes=True)
    if pause:
        await deploy.set_enabled("demo", False)
    await deploy.deploy_file(e.path, yes=True)
    assert e.cloud.flow["enabled"] is (not pause)


@pytest.mark.anyio
async def test_enable_ack_without_remote_activation_stays_pending(
    initial_cloud, monkeypatch
):
    e = initial_cloud
    with monkeypatch.context() as patch:
        patch.setattr(e.cloud, "enable_flow", AsyncMock(return_value={"enabled": True}))
        with pytest.raises(SandboxDeploymentError, match="activation"):
            await deploy.deploy_file(e.path, yes=True)
    await deploy.deploy_file(e.path, yes=True)
    assert e.cloud.flow["enabled"]


@pytest.mark.anyio
async def test_full_cleanup_closes_interrupted_replacement(initial_cloud):
    e = initial_cloud
    await deploy.deploy_file(e.path, yes=True)
    await interrupt_schedule(e)
    await deploy.cleanup_owned(e.store.read(), delete_function=True)
    await deploy.deploy_file(e.path, yes=True)
    assert (await deploy.set_enabled("demo", True))["enabled"]


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
        await deploy.deploy_file(Path("demo.py"), yes=True)
    corrected = replace(e.artifact, source="corrected", digest="corrected")
    result = await deploy.deploy_file(Path("demo.py"), artifact=corrected, yes=True)
    assert result.version == 3 and e.cloud.flow["enabled"]


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


@pytest.mark.anyio
async def test_replace_terminally_rejected_first_deploy(initial_cloud, monkeypatch):
    e = initial_cloud

    async def inspect(function_id, version, **kw):
        return {"version": version, "status": "rejected" if version == 1 else "ready"}

    monkeypatch.setattr(e.cloud, "get_version", inspect)
    with pytest.raises(SandboxDeploymentError, match="rejected"):
        await deploy.deploy_file(e.path, yes=True)
    assert not e.cloud.flow
    result = await deploy.deploy_file(
        e.path,
        yes=True,
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


@pytest.mark.anyio
@pytest.mark.parametrize("foreign_flow", [False, True])
async def test_legacy_initial_recovers_only_function_for_cleanup(
    initial_cloud, foreign_flow
):
    e = initial_cloud
    e.cloud.fail = "function"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(e.path, yes=True)
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
            yes=True,
            device_free=True,
        )
    assert e.cloud.create_calls == before
    with pytest.raises(SandboxDeploymentConflict, match="cleanup") as caught:
        await deploy.deploy_file(e.path, artifact=schedule, yes=True, device_free=True)
    assert e.cloud.create_calls == before * 2
    assert (
        caught.value.receipt == e.store.read()
        and caught.value.receipt.function_id == "f"
    )
    await deploy.cleanup_owned(e.store.read(), delete_function=True)
    assert not e.cloud.function_exists and e.cloud.flow == expected_flow
    assert set(e.cloud.calls) <= {"read_document", "delete_function"}


@pytest.mark.anyio
async def test_legacy_initial_uses_pinned_document(initial_cloud):
    e = initial_cloud
    e.cloud.fail = "flow"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(e.path, yes=True)
    remove_preview(e.store)
    before = list(e.cloud.create_calls)
    schedule = replace(
        e.artifact, declaration=CloudDeclaration("demo", schedule="* * * * *")
    )
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(e.path, artifact=schedule, yes=True, device_free=True)
    assert e.cloud.create_calls == before
    assert (await deploy.deploy_file(e.path, yes=True)).endpoint


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["ready", "pending"])
async def test_first_deploy_supersession_rechecks_rejection(
    initial_cloud, monkeypatch, status
):
    e = initial_cloud
    inspect = AsyncMock(return_value={"version": 1, "status": "rejected"})
    monkeypatch.setattr(e.cloud, "get_version", inspect)
    with pytest.raises(SandboxDeploymentError, match="rejected"):
        await deploy.deploy_file(e.path, yes=True)
    inspect.return_value = {"version": 1, "status": status}
    before = e.store.read()
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(
            e.path,
            yes=True,
            artifact=replace(e.artifact, source="corrected", digest="corrected"),
        )
    assert e.store.read() == before and e.cloud.version == 1 and not e.cloud.flow


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


@pytest.mark.anyio
async def test_initial_schema_const_requires_same_json_type(initial_cloud):
    e = initial_cloud
    e.cloud.fail = "function"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(
            e.path, yes=True, artifact=replace(e.artifact, input_schema={"const": True})
        )
    before = list(e.cloud.create_calls)
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(
            e.path, yes=True, artifact=replace(e.artifact, input_schema={"const": 1})
        )
    assert e.cloud.create_calls == before


@pytest.mark.anyio
async def test_cutover_schema_const_requires_same_json_type(changed):
    e = changed
    e.cloud.fail = "replace"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(
            Path("demo.py"),
            yes=True,
            artifact=replace(e.artifact, input_schema={"const": True}),
        )
    with pytest.raises(SandboxDeploymentConflict):
        await deploy.deploy_file(
            Path("demo.py"),
            yes=True,
            artifact=replace(e.artifact, input_schema={"const": 1}),
        )
    assert not e.cloud.flow["enabled"]


def test_remote_document_bool_is_not_number():
    assert not deploy.equivalent_document(
        {"on": {"custom": {"inputs": {"const": True}}}},
        {"on": {"custom": {"inputs": {"const": 1}}}},
    )


@pytest.mark.anyio
async def test_initial_receipt_accepts_equivalent_number(initial_cloud):
    e = initial_cloud
    e.cloud.fail = "function"
    with pytest.raises(SandboxDeploymentError):
        await deploy.deploy_file(
            e.path,
            yes=True,
            artifact=replace(e.artifact, input_schema={"minimum": 0.0}),
        )
    result = await deploy.deploy_file(
        e.path, yes=True, artifact=replace(e.artifact, input_schema={"minimum": 0})
    )
    assert result.flow_id and e.cloud.flow["enabled"]


def test_remote_document_accepts_equivalent_number():
    assert deploy.equivalent_document(
        {"on": {"custom": {"inputs": {"minimum": 0}}}, "version": 1},
        {"on": {"custom": {"inputs": {"minimum": 0.0}}}, "version": 1.0},
    )
