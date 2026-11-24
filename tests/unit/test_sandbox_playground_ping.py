from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from mobilerun.sandbox.cli.group import sandbox
from mobilerun.sandbox.deployment import build_document, equivalent_document
from mobilerun.sandbox.models import (
    Artifact,
    CloudDeclaration,
    DeploymentReceipt,
    Endpoint,
)
from tests.support.sandbox_playground import ping


def test_ping_is_constant_for_all_names():
    for name in (None, "someone"):
        assert ping.main(ping.Input(name=name), None).model_dump() == {
            "message": "pong",
            "version": 1,
        }


def test_device_free_document_preserves_payload_schema():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    artifact = Artifact(CloudDeclaration("playground-ping"), "", "", schema, None)
    doc = build_document(artifact, "function-id", 1, device=None)
    assert "devices" not in doc
    assert doc["on"]["custom"]["inputs"] == schema
    assert doc["steps"][0]["with"]["input"] == "{{event.payload}}"


def test_device_free_document_does_not_match_device_bound_flow():
    artifact = Artifact(CloudDeclaration("playground-ping"), "", "", {}, None)
    doc = build_document(artifact, "function-id", 1, device=None)
    assert not equivalent_document({**doc, "devices": ["fake-device"]}, doc)


@pytest.fixture
def endpoint_contract(monkeypatch):
    endpoint = Endpoint(
        "https://example.test/v1/triggers/trigger-id/fire", "trigger-id"
    )
    receipt = DeploymentReceipt(
        "playground-ping",
        "function-id",
        1,
        "created",
        "flow-id",
        "trigger-id",
        (),
        "digest",
        endpoint,
        (),
    )
    deploy_file = AsyncMock(return_value=receipt)
    from importlib import import_module

    monkeypatch.setattr(
        import_module("mobilerun.sandbox.deployment"), "deploy_file", deploy_file
    )
    result = CliRunner().invoke(sandbox, ["deploy", str(Path(ping.__file__)), "--yes"])
    assert result.exit_code == 0, result.output
    deploy_file.assert_awaited_once()
    return result.output


def test_contract_reports_sdk_trigger_and_flow(endpoint_contract):
    assert "/triggers/trigger-id/fire" in endpoint_contract
    assert "flow-id" in endpoint_contract


def test_contract_reports_auth_payload_and_invocation_poll(endpoint_contract):
    # TODO(Task 6/7): CLI deploy currently renders only the URL and flow ID.
    # Keep the remaining playground contract assertions pending CLI integration.
    expected = ("Bearer <YOUR_API_KEY>", '"payload": {}', "invocationId", "/executions")
    missing = [text for text in expected if text not in endpoint_contract]
    if len(missing) == len(expected):
        pytest.fail("CLI deploy did not render the auth/payload/poll contract")
    assert not missing
