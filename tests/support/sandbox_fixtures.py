"""Explicitly imported, function-scoped fixtures for sandbox tests."""

import importlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from mobilerun.sandbox import clients
from mobilerun.sandbox import deployment as deploy
from mobilerun.sandbox.models import Artifact, CloudDeclaration, OwnershipReceipt
from mobilerun.sandbox.settings import SandboxSettings
from tests.support.sandbox_fakes import InitialSandboxCloud, SandboxCloud

__all__ = [
    "remote",
    "env",
    "initial_cloud",
    "changed",
    "setup_deploy",
    "Input",
    "Output",
    "backend",
]


def execution(status="success", body=None):
    return {
        "id": "execution",
        "flowId": "flow",
        "invocationId": "invocation",
        "status": status,
        "startedAt": "2026-11-24T10:00:00Z",
        "finishedAt": "2026-11-24T10:01:00Z",
        "result": {
            "steps": [
                {
                    "uses": "functions.run",
                    "status": "success",
                    "body": (
                        body
                        if body is not None
                        else {"outcome": "complete", "result": {"n": 2}}
                    ),
                }
            ]
        },
    }


@pytest.fixture
def remote(tmp_path, monkeypatch):
    scope = deploy.Scope("https://example.test/v1", "fingerprint")
    settings = SandboxSettings(
        MOBILERUN_SANDBOX_DIRECTORY=str(tmp_path), MOBILERUN_SANDBOX_POLL_INTERVAL=0.001
    )
    monkeypatch.setattr(deploy, "current_scope", lambda: scope)
    monkeypatch.setattr(deploy, "get_sandbox_settings", lambda: settings)
    store = deploy.ReceiptStore(scope, "demo", settings)
    receipt = OwnershipReceipt(
        scope.origin,
        scope.account_id,
        "demo",
        "function",
        "flow",
        "trigger",
        ("action",),
        "digest",
        (),
    )
    store.write(receipt)
    document = {
        "steps": [
            {"uses": "functions.run", "with": {"functionId": "function", "version": 2}}
        ],
        "on": {
            "custom": {
                "inputs": {
                    "type": "object",
                    "required": ["n"],
                    "properties": {"n": {"type": "integer"}},
                    "additionalProperties": False,
                }
            }
        },
    }
    flow = {
        "id": "flow",
        "name": "demo · flow",
        "triggerId": "trigger",
        "enabled": True,
    }
    fake = SimpleNamespace(
        list_functions=AsyncMock(
            return_value=[
                {
                    "id": "function",
                    "name": "demo",
                    "latestVersion": {"version": 3, "status": "rejected"},
                    "apiKey": "PRIVATE",
                }
            ]
        ),
        list_versions=AsyncMock(
            return_value=[
                {"version": 3, "status": "rejected"},
                {"version": 2, "status": "ready"},
                {"version": 1, "status": "ready"},
            ]
        ),
        get_version=AsyncMock(
            return_value={
                "version": 2,
                "status": "ready",
                "inputSchema": document["on"]["custom"]["inputs"],
            }
        ),
        list_flows=AsyncMock(return_value=[flow]),
        list_actions=AsyncMock(return_value=[{"actionId": "action"}]),
        get_document=AsyncMock(return_value=document),
        fire_trigger=AsyncMock(
            return_value={"invocationId": "invocation", "enqueuedCount": 1}
        ),
        list_executions=AsyncMock(
            return_value=[
                {"id": "execution", "invocationId": "invocation", "flowId": "flow"}
            ]
        ),
        get_execution=AsyncMock(return_value=execution()),
        enable_flow=AsyncMock(return_value={"enabled": True}),
        disable_flow=AsyncMock(return_value={"enabled": False}),
        run_logs=AsyncMock(return_value="downloaded log"),
    )
    for name, value in vars(fake).items():
        monkeypatch.setattr(clients, name, value)
    return SimpleNamespace(
        fake=fake, store=store, receipt=receipt, document=document, flow=flow
    )


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


@pytest.fixture
def initial_cloud(env, monkeypatch):
    cloud = InitialSandboxCloud(
        deploy.build_document(env.artifact, "f", 1, device=None)
    )
    monkeypatch.setattr(deploy, "clients", cloud)
    env.cloud = cloud
    return env


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
