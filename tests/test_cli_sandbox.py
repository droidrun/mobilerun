"""Sandbox CLI behaviour using generated response types, without live calls."""

import json
from unittest.mock import Mock, call

import httpx
import pytest
from click.testing import CliRunner
from mobilerun_sdk import APIStatusError
from mobilerun_sdk.types.compute.sandbox_create_response import SandboxCreateResponse
from mobilerun_sdk.types.compute.sandbox_exec_response import SandboxExecResponse
from mobilerun_sdk.types.compute.sandbox_list_response import SandboxListResponse
from test_cli_function_cloud import sdk  # noqa: F401

from mobilerun.cli.main import cli

ID = "31fe8bac-cf2a-44af-af62-88aa6c73c0c1"


def invoke(*args):
    return CliRunner().invoke(cli, ["sandbox", *map(str, args)])


def sandbox():
    return SandboxCreateResponse(
        data=dict(
            id=ID,
            name="demo",
            status="running",
            allowedHosts=[],
            idleTimeoutS=60,
            createdAt="2026-11-24T10:00:00Z",
            expiresAt="2026-11-24T11:00:00Z",
            usedSeconds=0,
        )
    )


def command(**updates):
    return SandboxExecResponse(
        data={
            **dict(
                commandId="cmd-1",
                status="exited",
                stdout="1\n",
                stderr="",
                exitCode=0,
                startedAt="2026-11-24T10:00:00Z",
                timedOut=False,
                truncated=False,
            ),
            **updates,
        }
    )


def test_create_passes_options(sdk):  # noqa: F811
    sdk.client.compute.sandboxes.create.return_value = sandbox()
    result = invoke(
        "create",
        "--name",
        "demo",
        "--timeout",
        300,
        "--idle-timeout",
        60,
        "--env",
        "MODE=test",
        "-j",
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["id"] == ID
    sdk.client.compute.sandboxes.create.assert_called_once_with(
        name="demo", timeout_s=300, idle_timeout_s=60, env={"MODE": "test"}
    )


def test_create_invalid_env_never_calls_api(sdk):  # noqa: F811
    result = invoke("create", "--env", "BROKEN")
    assert result.exit_code == 2
    sdk.factory.assert_not_called()


def test_list_includes_stopped_and_paginates(sdk):  # noqa: F811
    item = sandbox().data.model_dump(by_alias=True)
    sdk.client.compute.sandboxes.list.side_effect = [
        SandboxListResponse(
            data={
                "items": [item],
                "pagination": {
                    "page": 1,
                    "pageSize": 1,
                    "total": 2,
                    "pages": 2,
                    "hasNext": True,
                    "hasPrev": False,
                },
            }
        ),
        SandboxListResponse(
            data={
                "items": [{**item, "status": "stopped"}],
                "pagination": {
                    "page": 2,
                    "pageSize": 1,
                    "total": 2,
                    "pages": 2,
                    "hasNext": False,
                    "hasPrev": True,
                },
            }
        ),
    ]
    result = invoke("list", "-j")
    assert result.exit_code == 0, result.output
    assert [item["status"] for item in json.loads(result.stdout)] == [
        "running",
        "stopped",
    ]
    assert sdk.client.compute.sandboxes.list.call_args_list == [
        call(page=1),
        call(page=2),
    ]


def test_show(sdk):  # noqa: F811
    sdk.client.compute.sandboxes.retrieve.return_value = sandbox()
    result = invoke("show", ID)
    assert result.exit_code == 0, result.output
    assert ID in result.stdout
    assert "running" in result.stdout
    sdk.client.compute.sandboxes.retrieve.assert_called_once_with(ID)


@pytest.mark.parametrize("exit_code", [0, 7, 127])
def test_exec_propagates_exit_and_streams(sdk, exit_code):  # noqa: F811
    sdk.client.compute.sandboxes.exec.return_value = command(
        exitCode=exit_code, stderr="diagnostic\n"
    )
    result = invoke("exec", ID, "python -c 'print(1)'", "--cwd", "/tmp", "--env", "A=b")
    assert result.exit_code == exit_code, result.output
    assert result.stdout == "1\n"
    assert result.stderr == "diagnostic\n"
    sdk.client.compute.sandboxes.exec.assert_called_once_with(
        ID,
        command="python -c 'print(1)'",
        cwd="/tmp",
        env={"A": "b"},
        timeout_s=60,
        wait_s=30,
    )


def test_exec_waits_and_emits_one_json_document(sdk):  # noqa: F811
    sdk.client.compute.sandboxes.exec.return_value = command(
        status="running", exitCode=None, stdout=""
    )
    sdk.client.compute.sandboxes.commands.retrieve.return_value = command(exitCode=7)
    result = invoke("exec", ID, "exit 7", "-j")
    assert result.exit_code == 7, result.output
    assert json.loads(result.stdout)["exitCode"] == 7
    sdk.client.compute.sandboxes.commands.retrieve.assert_called_once_with(
        "cmd-1", sandbox_id=ID, wait_s=5
    )


def test_exec_timeout_without_exit_code_fails(sdk):  # noqa: F811
    sdk.client.compute.sandboxes.exec.return_value = command(
        exitCode=None, timedOut=True
    )
    result = invoke("exec", ID, "sleep 999")
    assert result.exit_code == 1
    assert "timed out" in result.stderr


def test_files_upload_download_exact_bytes(sdk, tmp_path):  # noqa: F811
    source = tmp_path / "source.bin"
    source.write_bytes(b"\x00\xffpayload")
    payloads = []
    sdk.client.compute.sandboxes.files.upload.side_effect = (
        lambda sandbox_id, source, **kwargs: payloads.append(source.read_bytes())
    )
    uploaded = invoke("files", "upload", ID, source, "/tmp/data", "-j")
    assert uploaded.exit_code == 0, uploaded.output
    assert json.loads(uploaded.stdout) == {"path": "/tmp/data"}
    assert payloads == [b"\x00\xffpayload"]
    assert sdk.client.compute.sandboxes.files.upload.call_args.args == (ID, source)
    assert sdk.client.compute.sandboxes.files.upload.call_args.kwargs == {
        "path": "/tmp/data"
    }
    sdk.client.compute.sandboxes.files.retrieve.return_value = httpx.Response(
        200, content=b"\x00\xffpayload"
    )
    target = tmp_path / "download.bin"
    downloaded = invoke("files", "download", ID, "/tmp/data", "-o", target, "-j")
    assert downloaded.exit_code == 0, downloaded.output
    assert target.read_bytes() == b"\x00\xffpayload"
    sdk.client.compute.sandboxes.files.retrieve.assert_called_once_with(
        ID, path="/tmp/data"
    )


def test_download_refuses_overwrite_before_network(sdk, tmp_path):  # noqa: F811
    target = tmp_path / "existing"
    target.write_bytes(b"keep")
    result = invoke("files", "download", ID, "/tmp/data", "-o", target)
    assert result.exit_code == 1
    assert target.read_bytes() == b"keep"
    sdk.factory.assert_not_called()


@pytest.mark.parametrize("verb", ["stop", "delete"])
def test_stop_delete_warn_about_retained_record(sdk, verb):  # noqa: F811
    result = invoke(verb, ID, "--yes", "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"id": ID, "stopped": True}
    assert result.stderr == ""
    human = invoke(verb, ID, "--yes")
    assert human.exit_code == 0, human.output
    assert human.stderr.count("still appear as stopped") == 1
    assert sdk.client.compute.sandboxes.delete.call_args_list == [call(ID), call(ID)]


@pytest.mark.parametrize("status", [401, 403, 404, 409, 422, 429])
def test_sandbox_api_errors_are_actionable_and_redacted(sdk, status):  # noqa: F811
    request = httpx.Request("GET", "https://example.test/v1/compute/sandboxes/missing")
    sdk.client.compute.sandboxes.retrieve.side_effect = APIStatusError(
        "error",
        response=httpx.Response(status, request=request),
        body={"message": "Bad unit-cloud-function-key"},
    )
    result = invoke("show", ID, "-j")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == status
    assert "unit-cloud-function-key" not in result.output
    assert "[redacted]" in result.stderr
    assert len(result.stderr.splitlines()) >= 2


@pytest.mark.parametrize(
    "verb,args,method",
    [
        ("create", [], "create"),
        ("list", [], "list"),
        ("exec", [ID, "true"], "exec"),
        ("stop", [ID, "--yes"], "delete"),
        ("delete", [ID, "--yes"], "delete"),
        ("files", ["download", ID, "missing", "-o", "download.bin"], "files.retrieve"),
    ],
)
def test_sandbox_operation_failure_is_nonzero(
    sdk, tmp_path, monkeypatch, verb, args, method  # noqa: F811
):
    from operator import attrgetter

    monkeypatch.chdir(tmp_path)
    error = APIStatusError(
        "missing",
        response=httpx.Response(
            404,
            request=httpx.Request(
                "GET", "https://example.test/v1/compute/sandboxes/missing"
            ),
        ),
        body={"message": "Sandbox not found"},
    )
    attrgetter(method)(sdk.client.compute.sandboxes).side_effect = error
    result = invoke(verb, *args, "-j")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == 404
    assert "Sandbox not found" in result.stderr


def test_upload_sdk_retry_resends_complete_content(
    sdk, tmp_path, monkeypatch  # noqa: F811
):
    import time

    from mobilerun_sdk import Mobilerun

    source = tmp_path / "payload.bin"
    source.write_bytes(b"\x00\xffcomplete payload")
    bodies = []
    responses = iter([502, 204])

    def handle(request):
        bodies.append(request.read())
        return httpx.Response(next(responses), request=request)

    monkeypatch.setattr(time, "sleep", lambda delay: None)
    sdk.factory.return_value = Mobilerun(
        api_key="unit-cloud-function-key",
        max_retries=1,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    result = invoke("files", "upload", ID, source, "payload.bin", "-j")
    assert result.exit_code == 0, result.output
    assert bodies == [b"\x00\xffcomplete payload", b"\x00\xffcomplete payload"]


def test_exec_still_running_identifies_background_command(
    sdk, monkeypatch  # noqa: F811
):
    from mobilerun.cli import sandbox as module

    sdk.client.compute.sandboxes.exec.return_value = command(
        status="running", exitCode=None
    )
    sdk.client.compute.sandboxes.commands.retrieve.return_value = command(
        status="running", exitCode=None
    )
    monkeypatch.setattr(module.time, "monotonic", Mock(side_effect=[0, 0, 32]))
    result = invoke("exec", ID, "long-running-command", "--timeout", "1", "-j")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["commandId"] == "cmd-1"
    assert "Command cmd-1 is still running and may still finish" in result.stderr


def test_sandbox_tsv_uses_iso_timestamp(sdk):  # noqa: F811
    sdk.client.compute.sandboxes.retrieve.return_value = sandbox()
    result = invoke("show", ID)
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{ID}\tdemo\trunning\t2026-11-24T11:00:00+00:00\n"
