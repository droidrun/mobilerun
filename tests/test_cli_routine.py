"""Routine document commands against a fake generated SDK client, without network."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from click.testing import CliRunner
from mobilerun_sdk import APIStatusError
from mobilerun_sdk.types.workflows.flow_validate_document_response import (
    Data as Verdict,
)
from test_cli_script_cloud import sdk  # noqa: F401

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
    assert "created disabled; activate it in Mobilerun Cloud" in result.stderr
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
    sdk, document_file, json_output
):  # noqa: F811
    body = {"code": "document_invalid", "message": "Invalid", "details": ERRORS}
    sdk.client.workflows.flows.create_from_document.side_effect = status_error(
        422, body
    )
    result = invoke("push", document_file, *(["-j"] if json_output else []))
    assert result.exit_code == 3, result.output
    assert result.stderr == ISSUES_TEXT
    assert result.stdout == (json.dumps(body) + "\n" if json_output else "")


def test_pull_stdout(sdk):  # noqa: F811
    sdk.client.workflows.flows.retrieve_document.return_value = SimpleNamespace(
        data=DOCUMENT
    )
    result = invoke("pull", FLOW_ID)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == DOCUMENT
    sdk.client.workflows.flows.retrieve_document.assert_called_once_with(FLOW_ID)


def test_pull_writes_file_and_refuses_overwrite(sdk, tmp_path):  # noqa: F811
    sdk.client.workflows.flows.retrieve_document.return_value = SimpleNamespace(
        data=DOCUMENT
    )
    target = tmp_path / "out.json"
    first = invoke("pull", FLOW_ID, "-o", target)
    second = invoke("pull", FLOW_ID, "-o", target)
    forced = invoke("pull", FLOW_ID, "-o", target, "--force")
    assert first.exit_code == forced.exit_code == 0, first.output + forced.output
    assert second.exit_code == 1, second.output
    assert "already exists; use --force" in second.stderr
    assert json.loads(target.read_text()) == DOCUMENT


def page(*items, has_next=False):
    return SimpleNamespace(
        items=list(items), pagination=SimpleNamespace(has_next=has_next)
    )


def test_list_collects_all_pages(sdk):  # noqa: F811
    second_id = "31fe8bac-cf2a-44af-af62-88aa6c73c0b2"
    sdk.client.workflows.flows.list.side_effect = [
        page(flow(), has_next=True),
        page(flow(id=second_id, name="Evening", lifecycle_status="enabled")),
    ]
    text = invoke("list")
    sdk.client.workflows.flows.list.side_effect = [
        page(flow(), has_next=True),
        page(flow(id=second_id, name="Evening", lifecycle_status="enabled")),
    ]
    data = invoke("list", "-j")
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
