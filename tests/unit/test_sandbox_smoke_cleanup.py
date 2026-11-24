"""Exercise the smoke test's persisted-receipt cleanup with a fake cloud."""

import subprocess
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.models.billing import Billing
from mobilerun.sandbox.errors import SandboxDeploymentConflict, SandboxDeploymentError
from mobilerun.sandbox.models import Artifact, CloudDeclaration, OwnershipReceipt
from tests.support.sandbox_fakes import InitialSandboxCloud


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
    scope = deployment.Scope("https://example.test/v1", "a" * 64)
    monkeypatch.setenv("MOBILERUN_SANDBOX_DIRECTORY", str(tmp_path / "sandbox"))
    monkeypatch.setattr(deployment, "current_scope", lambda: scope)
    monkeypatch.setattr(deployment, "package", lambda path: artifact)
    document = deployment.build_document(artifact, "f", 1, device=None)
    cloud = InitialSandboxCloud(document)
    monkeypatch.setattr(deployment, "clients", cloud)
    return deployment, source, deployment.ReceiptStore(scope, name), cloud


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
        await deployment.deploy_file(source, billing=Billing.PAID, yes=True)

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
        await deployment.cleanup_owned(
            receipt, delete_function=True, billing=Billing.NO_COMMERCE
        )
    assert cloud.calls == before
    assert store.read() == receipt

    # Only in the fake: replay the original intent to recover the missing IDs.
    await deployment.deploy_file(source, billing=Billing.PAID, yes=True)
    await deployment.cleanup_owned(
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
        await deployment.deploy_file(source, billing=Billing.PAID, yes=True)
    receipt = store.read()
    assert isinstance(receipt, OwnershipReceipt)
    assert receipt.function_id == "f"
    assert receipt.flow_id == ("flow" if failed_at == "enable" else None)

    await deployment.cleanup_owned(
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
        await deployment.deploy_file(source, billing=Billing.PAID, yes=True)
    assert store.read() is None
    assert cloud.function_exists and not cloud.calls and not cloud.create_calls


@pytest.mark.anyio
async def test_unattached_foreign_action_retains_owned_function(setup):
    deployment, source, store, cloud = setup
    await deployment.deploy_file(source, yes=True)
    cloud.actions["foreign"] = {
        "id": "foreign",
        "uses": "functions.run",
        "params": {"functionId": "f", "version": 1},
    }
    with pytest.raises(SandboxDeploymentError) as error:
        await deployment.cleanup_owned(
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
    await deployment.deploy_file(source, yes=True)

    async def definitions(**kwargs):
        if definition is None:
            raise RuntimeError("action listing unavailable")
        return [definition]

    monkeypatch.setattr(cloud, "list_action_definitions", definitions, raising=False)
    with pytest.raises(SandboxDeploymentError):
        await deployment.cleanup_owned(store.read(), delete_function=True)
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
    await deployment.deploy_file(source, yes=True)
    cloud.actions["foreign"] = definition
    await deployment.cleanup_owned(store.read(), delete_function=True)
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
    root = Path(__file__).parents[2]
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
        cwd=root,
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
    deployment = import_module("mobilerun.sandbox.deployment")
    expected = "https://staging.example.test/v1"
    monkeypatch.setenv("MOBILERUN_CLOUD_BASE_URL", expected)
    scope = deployment.Scope(expected, "fingerprint")
    monkeypatch.setattr(deployment, "current_scope", lambda: scope)
    assert smoke._verified_smoke_scope() == scope
    monkeypatch.setattr(
        deployment,
        "current_scope",
        lambda: deployment.Scope("https://other.example.test/v1", "fingerprint"),
    )
    with pytest.raises(AssertionError, match="origin"):
        smoke._verified_smoke_scope()
