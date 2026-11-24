"""Routine document commands against a fake generated SDK client, without network."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import call

import httpx
import pytest
from click.testing import CliRunner
from mobilerun_sdk import APIStatusError
from mobilerun_sdk.types.workflows.flow_validate_document_response import (
    Data as Verdict,
)
from test_cli_function_cloud import sdk  # noqa: F401

from mobilerun.cli.main import cli

FLOW_ID = "31fe8bac-cf2a-44af-af62-88aa6c73c0b1"
DOCUMENT = {"name": "Morning check", "steps": []}
SCHEMA = {"type": "object", "title": "Routine"}
REPAIR = {"action": "fix_spec_and_restart", "hint": "Add a step"}
ERRORS = [
    {
        "code": "document_invalid",
        "field": "/steps",
        "message": "At least one step is required",
        "repair": REPAIR,
    },
    {
        "code": "template_reference_invalid",
        "field": "",
        "message": "Unknown reference",
        "repair": REPAIR,
    },
]
ISSUES_TEXT = (
    "/steps: At least one step is required — Add a step\n"
    "(document): Unknown reference — Add a step\n"
)


def invoke(*args, **kwargs):
    return CliRunner().invoke(cli, ["routine", *map(str, args)], **kwargs)


def verdict(valid, errors=(), warnings=()):
    return SimpleNamespace(
        data=Verdict(valid=valid, errors=list(errors), warnings=list(warnings))
    )


def flow(**updates):
    values = dict(
        id=FLOW_ID,
        name="Morning check",
        lifecycle_status="disabled",
        updated_at=datetime(2026, 11, 16, tzinfo=timezone.utc),
    )
    values.update(updates)
    return SimpleNamespace(
        **values,
        model_dump=lambda **kwargs: {"id": values["id"], "name": values["name"]},
    )


def written(flow_value):
    return SimpleNamespace(
        data=SimpleNamespace(
            flow=flow_value,
            document=DOCUMENT,
            model_dump=lambda **kwargs: {"flow": {"id": flow_value.id}},
        )
    )


@pytest.fixture
def document_file(tmp_path):
    path = tmp_path / "routine.json"
    path.write_text(json.dumps(DOCUMENT))
    return path


def test_schema_stdout_and_file(sdk, tmp_path):  # noqa: F811
    sdk.client.workflows.flows.document_schema.return_value = SCHEMA
    printed = invoke("schema")
    target = tmp_path / "schema.json"
    saved = invoke("schema", "-o", target)
    assert printed.exit_code == saved.exit_code == 0, printed.output + saved.output
    assert json.loads(printed.stdout) == SCHEMA
    assert json.loads(target.read_text()) == SCHEMA
    assert saved.stdout == f"{target}\n"


def test_check_non_json_file_never_calls_the_api(sdk, tmp_path):  # noqa: F811
    path = tmp_path / "routine.json"
    path.write_text("not json")
    result = invoke("check", path)
    assert result.exit_code == 1, result.output
    assert "as JSON" in result.stderr
    sdk.factory.assert_not_called()


def test_check_valid_with_warnings(sdk, document_file):  # noqa: F811
    sdk.client.workflows.flows.validate_document.return_value = verdict(
        True, warnings=[{"code": "slow", "message": "Runs long"}]
    )
    result = invoke("check", document_file)
    assert result.exit_code == 0, result.output
    assert result.stdout == "valid\n"
    assert result.stderr == "warning: Runs long\n"
    sdk.client.workflows.flows.validate_document.assert_called_once_with(
        document=DOCUMENT
    )


def test_check_invalid_renders_issues(sdk, document_file):  # noqa: F811
    sdk.client.workflows.flows.validate_document.return_value = verdict(False, ERRORS)
    result = invoke("check", document_file)
    assert result.exit_code == 3, result.output
    assert result.stdout == ""
    assert result.stderr == ISSUES_TEXT


def test_check_json_is_server_document(sdk, document_file):  # noqa: F811
    sdk.client.workflows.flows.validate_document.return_value = verdict(False, ERRORS)
    result = invoke("check", document_file, "-j")
    assert result.exit_code == 3, result.output
    document = json.loads(result.stdout)
    assert document["valid"] is False
    assert document["errors"][0]["field"] == "/steps"


def test_push_creates_disabled_routine(sdk, document_file):  # noqa: F811
    sdk.client.workflows.flows.create_from_document.return_value = written(flow())
    result = invoke("push", document_file)
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{FLOW_ID}\tMorning check\tdisabled\n"
    assert "created disabled; enable it with" in result.stderr
    sdk.client.workflows.flows.create_from_document.assert_called_once_with(
        document=DOCUMENT
    )
    sdk.client.workflows.flows.replace_document.assert_not_called()


def test_push_with_id_replaces(sdk, document_file):  # noqa: F811
    sdk.client.workflows.flows.replace_document.return_value = written(flow())
    result = invoke("push", document_file, "--id", FLOW_ID)
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    sdk.client.workflows.flows.replace_document.assert_called_once_with(
        FLOW_ID, document=DOCUMENT
    )
    sdk.client.workflows.flows.create_from_document.assert_not_called()


def status_error(status, body, headers=None):
    request = httpx.Request("POST", "https://example.test/v1/workflows/flows/document")
    return APIStatusError(
        "error",
        response=httpx.Response(status, request=request, headers=headers or {}),
        body=body,
    )


@pytest.mark.parametrize("json_output", [False, True])
def test_push_rejected_document_renders_issues(
    sdk, document_file, json_output  # noqa: F811
):
    body = {"code": "document_invalid", "message": "Invalid", "details": ERRORS}
    sdk.client.workflows.flows.create_from_document.side_effect = status_error(
        422, body
    )
    result = invoke("push", document_file, *(["-j"] if json_output else []))
    assert result.exit_code == 3, result.output
    assert result.stderr == ISSUES_TEXT
    if json_output:
        assert json.loads(result.stdout) == {
            "error": "Invalid",
            "status": 422,
            "code": "document_invalid",
            "details": ERRORS,
        }
    else:
        assert result.stdout == ""


def test_push_rejected_document_redacts_secrets(sdk, document_file):  # noqa: F811
    leaky = [{**ERRORS[0], "message": "Bad value unit-cloud-function-key"}]
    body = {"code": "document_invalid", "message": "Invalid", "details": leaky}
    sdk.client.workflows.flows.create_from_document.side_effect = status_error(
        422, body
    )
    result = invoke("push", document_file, "-j")
    assert result.exit_code == 3, result.output
    assert "unit-cloud-function-key" not in result.output
    assert "[redacted]" in result.stdout
    assert "[redacted]" in result.stderr


def test_pull_stdout(sdk):  # noqa: F811
    sdk.client.workflows.flows.retrieve_document.return_value = SimpleNamespace(
        data=DOCUMENT
    )
    result = invoke("pull", FLOW_ID, "--format", "json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == DOCUMENT
    sdk.client.workflows.flows.retrieve_document.assert_called_once_with(FLOW_ID)


def test_pull_writes_file_and_refuses_overwrite(sdk, tmp_path):  # noqa: F811
    retrieve = sdk.client.workflows.flows.retrieve_document
    retrieve.return_value = SimpleNamespace(data=DOCUMENT)
    target = tmp_path / "out.json"
    first = invoke("pull", FLOW_ID, "--format", "json", "-o", target)
    first_bytes = target.read_bytes()
    assert first.exit_code == 0, first.output
    assert json.loads(first_bytes) == DOCUMENT
    retrieve.return_value = SimpleNamespace(data={**DOCUMENT, "name": "Changed"})
    second = invoke("pull", FLOW_ID, "--format", "json", "-o", target)
    assert second.exit_code == 1, second.output
    assert "already exists; use --force" in second.stderr
    assert target.read_bytes() == first_bytes
    forced = invoke("pull", FLOW_ID, "--format", "json", "-o", target, "--force")
    assert forced.exit_code == 0, forced.output
    assert json.loads(target.read_bytes()) == {**DOCUMENT, "name": "Changed"}


@pytest.mark.parametrize(
    "args,call",
    [(("pull", FLOW_ID), "retrieve_document"), (("schema",), "document_schema")],
)
def test_existing_output_is_refused_before_the_download(
    sdk, tmp_path, args, call  # noqa: F811
):
    target = tmp_path / "out.json"
    target.write_text("keep")
    result = invoke(*args, "-o", target)
    assert result.exit_code == 1, result.output
    assert "already exists; use --force" in result.stderr
    assert target.read_text() == "keep"
    sdk.factory.assert_not_called()
    getattr(sdk.client.workflows.flows, call).assert_not_called()


def page(*items, has_next=False):
    return SimpleNamespace(
        items=list(items), pagination=SimpleNamespace(has_next=has_next)
    )


def test_list_collects_all_pages(sdk):  # noqa: F811
    second_id = "31fe8bac-cf2a-44af-af62-88aa6c73c0b2"
    pages = [
        page(flow(), has_next=True),
        page(flow(id=second_id, name="Evening", lifecycle_status="enabled")),
    ]
    listing = sdk.client.workflows.flows.list
    listing.side_effect = pages
    text = invoke("list")
    assert listing.call_args_list == [call(page=1), call(page=2)]
    listing.reset_mock()
    listing.side_effect = pages
    data = invoke("list", "-j")
    assert listing.call_args_list == [call(page=1), call(page=2)]
    assert text.exit_code == data.exit_code == 0, text.output + data.output
    assert text.stdout == (
        f"{FLOW_ID}\tMorning check\tdisabled\t2026-11-16T00:00:00+00:00\n"
        f"{second_id}\tEvening\tenabled\t2026-11-16T00:00:00+00:00\n"
    )
    assert [item["id"] for item in json.loads(data.stdout)] == [FLOW_ID, second_id]


@pytest.mark.parametrize(
    "status,message",
    [(401, "Unauthorized"), (404, "Routine not found"), (429, "Slow down")],
)
def test_api_errors_use_the_shared_converter(sdk, status, message):  # noqa: F811
    sdk.client.workflows.flows.retrieve_document.side_effect = status_error(
        status, {"message": message}
    )
    result = invoke("pull", FLOW_ID)
    assert result.exit_code == 1, result.output
    assert f"(HTTP {status})" in result.stderr
    assert message in result.stderr


def test_yaml_12_values_are_sent_as_json(sdk, tmp_path):  # noqa: F811
    from mobilerun_sdk.types.workflows.flow_validate_document_response import (
        FlowValidateDocumentResponse,
    )

    path = tmp_path / "routine.yaml"
    path.write_text(
        "name: demo\non: {custom: {}}\nvalues: {switch: on, answer: yes, date: 2026-11-24, count: 012, exponent: 1e3, octal: 0o12, enabled: true}\nsteps: []\n"
    )
    sdk.client.workflows.flows.validate_document.return_value = (
        FlowValidateDocumentResponse(data={"valid": True, "errors": [], "warnings": []})
    )
    result = invoke("check", path, "-j")
    assert result.exit_code == 0, result.output
    sdk.client.workflows.flows.validate_document.assert_called_once_with(
        document={
            "name": "demo",
            "on": {"custom": {}},
            "steps": [],
            "values": {
                "switch": "on",
                "answer": "yes",
                "date": "2026-11-24",
                "count": 12,
                "exponent": 1000.0,
                "octal": 10,
                "enabled": True,
            },
        }
    )


def test_document_yaml_json_roundtrip_and_schema_first(sdk, tmp_path):  # noqa: F811
    document = {
        "name": "demo",
        "on": {"custom": {}},
        "$schema": "https://example.test/schema",
        "values": {"number": "1e3", "date": "2026-11-24", "enabled": "on", "count": 12},
        "steps": [],
    }
    sdk.client.workflows.flows.retrieve_document.return_value.data = document
    target = tmp_path / "routine.yaml"
    pulled = invoke("pull", FLOW_ID, "-o", target)
    assert pulled.exit_code == 0, pulled.output
    assert (
        target.read_text().splitlines()[0] == '"$schema": "https://example.test/schema"'
    )
    sdk.client.workflows.flows.create_from_document.return_value = written(flow())
    pushed = invoke("push", target, "-j")
    assert pushed.exit_code == 0, pushed.output
    sdk.client.workflows.flows.create_from_document.assert_called_once_with(
        document=document
    )


@pytest.mark.parametrize(
    "source",
    [
        "values: {bad: .nan}",
        "1: value",
        "- one\n- two",
        "name: [",
        "name: first\nname: second",
    ],
)
def test_yaml_non_json_values_fail_before_network(sdk, tmp_path, source):  # noqa: F811
    path = tmp_path / "bad.yaml"
    path.write_text(source)
    result = invoke("check", path)
    assert result.exit_code == 1
    assert "Cannot read" in result.stderr
    sdk.factory.assert_not_called()


@pytest.mark.parametrize(
    "verb,status", [("enable", "enabled"), ("disable", "disabled")]
)
def test_lifecycle_is_decided_by_api(sdk, verb, status):  # noqa: F811
    from mobilerun_sdk.types.workflows.flow_update_response import FlowUpdateResponse

    sdk.client.workflows.flows.update.return_value = FlowUpdateResponse.model_construct(
        data=flow()
    )
    result = invoke(verb, FLOW_ID, "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["id"] == FLOW_ID
    sdk.client.workflows.flows.update.assert_called_once_with(
        FLOW_ID, lifecycle_status=status
    )


def test_show_routine_metadata(sdk):  # noqa: F811
    from mobilerun_sdk.types.workflows.flow_retrieve_response import (
        FlowRetrieveResponse,
    )

    sdk.client.workflows.flows.retrieve.return_value = (
        FlowRetrieveResponse.model_construct(data=flow())
    )
    result = invoke("show", FLOW_ID, "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["id"] == FLOW_ID
    sdk.client.workflows.flows.retrieve.assert_called_once_with(FLOW_ID)


def test_run_sends_payload_and_prints_execution_ids(sdk):  # noqa: F811
    from mobilerun_sdk.types.workflows.flow_run_response import FlowRunResponse

    sdk.client.workflows.flows.run.return_value = FlowRunResponse(
        data={"enqueuedCount": 1, "executionIds": ["execution-1"]}
    )
    result = invoke("run", FLOW_ID, "--input", '{"name":"Ada"}', "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["executionIds"] == ["execution-1"]
    sdk.client.workflows.flows.run.assert_called_once_with(
        FLOW_ID, payload={"name": "Ada"}
    )


def test_run_skipped_is_a_failure(sdk):  # noqa: F811
    from mobilerun_sdk.types.workflows.flow_run_response import FlowRunResponse

    sdk.client.workflows.flows.run.return_value = FlowRunResponse(
        data={
            "enqueuedCount": 0,
            "executionIds": [],
            "skipped": {"reason": "flow_disabled"},
        }
    )
    result = invoke("run", FLOW_ID, "-j")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["skipped"]["reason"] == "flow_disabled"
    assert "flow_disabled" in result.stderr


@pytest.mark.parametrize("input_json", ["{", "[]"])
def test_run_rejects_non_object_payload(sdk, input_json):  # noqa: F811
    result = invoke("run", FLOW_ID, "--input", input_json)
    assert result.exit_code == 2
    sdk.factory.assert_not_called()


def test_runs_paginates(sdk):  # noqa: F811
    from mobilerun_sdk.types.workflows.execution_list_response import (
        ExecutionListResponse,
        Item,
    )

    item = Item.model_construct(id="execution-1", status="success", started_at=None)
    sdk.client.workflows.executions.list.side_effect = [
        ExecutionListResponse.model_construct(
            items=[item], pagination=SimpleNamespace(has_next=True)
        ),
        ExecutionListResponse.model_construct(
            items=[], pagination=SimpleNamespace(has_next=False)
        ),
    ]
    result = invoke("runs", FLOW_ID, "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)[0]["status"] == "success"
    assert sdk.client.workflows.executions.list.call_args_list == [
        call(flow_id=FLOW_ID, page=1),
        call(flow_id=FLOW_ID, page=2),
    ]


def test_delete_requires_confirmation_then_deletes(sdk):  # noqa: F811
    denied = invoke("delete", FLOW_ID)
    assert denied.exit_code == 2
    sdk.factory.assert_not_called()
    result = invoke("delete", FLOW_ID, "--yes", "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"id": FLOW_ID, "deleted": True}
    sdk.client.workflows.flows.delete.assert_called_once_with(FLOW_ID)


@pytest.mark.parametrize(
    "verb,args,method,status",
    [
        ("schema", [], "flows.document_schema", 403),
        ("list", [], "flows.list", 429),
        ("show", [FLOW_ID], "flows.retrieve", 404),
        ("runs", [FLOW_ID], "executions.list", 404),
        ("enable", [FLOW_ID], "flows.update", 409),
        ("disable", [FLOW_ID], "flows.update", 404),
        ("delete", [FLOW_ID, "--yes"], "flows.delete", 404),
    ],
)
def test_routine_failures_propagate(sdk, verb, args, method, status):  # noqa: F811
    from operator import attrgetter

    attrgetter(method)(sdk.client.workflows).side_effect = status_error(
        status, {"message": "Request failed"}
    )
    result = invoke(verb, *args, "-j")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == status
    assert f"(HTTP {status})" in result.stderr


def test_run_enqueue_failure_is_not_success(sdk):  # noqa: F811
    from mobilerun_sdk.types.workflows.flow_run_response import FlowRunResponse

    sdk.client.workflows.flows.run.return_value = FlowRunResponse(
        data={"enqueuedCount": 0, "executionIds": [], "enqueueFailed": True}
    )
    result = invoke("run", FLOW_ID, "-j")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["enqueueFailed"] is True
    assert "could not enqueue" in result.stderr


@pytest.mark.parametrize(
    "suffix,source,message",
    [
        (".json", '{"name":"first","name":"second"}', "Duplicate document key: name"),
        (".json", '{"steps":[{"with":{"x":1,"x":2}}]}', "Duplicate document key: x"),
        (".yaml", "name: &name demo", "YAML anchors and aliases are not supported"),
        (".yaml", "name: *missing", "YAML anchors and aliases are not supported"),
        (
            ".yaml",
            "a: &a [1]\nb: [*a, *a]",
            "YAML anchors and aliases are not supported",
        ),
    ],
)
def test_documents_reject_ambiguous_or_expanding_values(
    sdk, tmp_path, suffix, source, message  # noqa: F811
):
    path = tmp_path / ("routine" + suffix)
    path.write_text(source)
    result = invoke("check", path, "-j")
    assert result.exit_code == 1
    assert message in json.loads(result.stdout)["error"]
    sdk.factory.assert_not_called()
