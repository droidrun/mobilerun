"""Cloud function contracts against a fake generated SDK client, without network."""

import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, call

import httpx
import pytest
from click.testing import CliRunner
from mobilerun_core.connection import cloud
from mobilerun_sdk import (
    APIStatusError,
)
from mobilerun_sdk.types.compute.function_source import FunctionSource
from mobilerun_sdk.types.compute.function_source_response import FunctionSourceResponse
from mobilerun_sdk.types.compute.functions.function_validation import FunctionValidation
from mobilerun_sdk.types.compute.serverless_function import ServerlessFunction

from mobilerun.cli import deviceauth, function, selection
from mobilerun.cli.main import cli
from mobilerun.config_manager import ConfigLoader

ID1 = "31fe8bac-cf2a-44af-af62-88aa6c73c0a1"
ID2 = "31fe8bac-cf2a-44af-af62-88aa6c73c0a2"
SOURCE = '# café\r\nprint("hello")\r\n'
DIGEST = hashlib.sha256(SOURCE.encode()).hexdigest()


def record(**updates):
    from mobilerun_sdk.types.compute.serverless_function import LatestVersion

    version = dict(
        apiKey={"full": True},
        functionId=ID2,
        version=2,
        createdAt="2026-11-13T00:00:00Z",
        manifest={"hosts": ["example.com"], "credentials": [], "scopes": []},
        inputSchema={"type": "object"},
        outputSchema={"type": "string"},
        sha256=DIGEST,
        sizeBytes=len(SOURCE.encode()),
        status="ready",
    )
    values = dict(
        id=ID2,
        name="hello.py",
        versionCount=2,
        createdAt="2026-11-13T00:00:00Z",
        updatedAt="2026-11-13T00:00:00Z",
        latestReadyVersion=2,
    )
    version.update(
        {
            key: value
            for key, value in updates.items()
            if key in version or key == "inspectError"
        }
    )
    values.update(
        {
            key: value
            for key, value in updates.items()
            if key not in version and key != "inspectError"
        }
    )
    values["latestVersion"] = LatestVersion(**version)
    return ServerlessFunction(**values)


def validation(**updates):
    values = dict(
        id=ID1,
        createdAt=datetime(2026, 11, 16, tzinfo=timezone.utc),
        status="done",
        valid=True,
        errors=[],
        warnings=[],
        outputKind="value",
        datasetKey=None,
        inputSchema={"type": "object"},
        outputSchema={"type": "string"},
    )
    values.update(updates)
    return FunctionValidation(**values)


def page(*items, has_next=False):
    return SimpleNamespace(
        data=SimpleNamespace(
            items=list(items), pagination=SimpleNamespace(has_next=has_next)
        )
    )


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ConfigLoader, "get_user_config_path", lambda: tmp_path / "config.yaml"
    )
    for name in (
        "MOBILERUN_CONFIG",
        "MOBILERUN_DEBUG",
        "MOBILERUN_API_BASE_URL",
        "MOBILERUN_CLOUD_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MOBILERUN_API_KEY", "unit-cloud-function-key")
    monkeypatch.setattr(deviceauth, "load_credentials", lambda: None)
    client = Mock()
    client.with_options.return_value = client
    factory = Mock(return_value=client)
    monkeypatch.setattr(cloud, "make_sdk_client", factory)
    client.compute.functions.create.return_value = SimpleNamespace(data=record())
    client.compute.functions.retrieve.return_value = SimpleNamespace(data=record())
    client.compute.functions.list.return_value = page(record())
    client.compute.functions.source.return_value = FunctionSourceResponse(
        data=FunctionSource(
            functionId=ID2, sha256=DIGEST, sizeBytes=len(SOURCE.encode()), source=SOURCE
        )
    )
    return SimpleNamespace(
        client=client, functions=client.compute.functions, factory=factory
    )


@pytest.fixture
def file(tmp_path):
    path = tmp_path / "hello.py"
    path.write_bytes(SOURCE.encode())
    return path


def invoke(*args, **kwargs):
    return CliRunner().invoke(cli, ["function", *map(str, args)], **kwargs)


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=0, sleeps=[])

    def sleep(seconds):
        state.sleeps.append(seconds)
        state.now += seconds

    monkeypatch.setattr(function.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(function.time, "sleep", sleep)
    return state


@pytest.mark.parametrize(
    "status,code",
    [
        (409, "function_limit_reached"),
        (409, "function_row_limit_reached"),
        (409, "function_busy"),
        (422, "function_name_invalid"),
        (422, "manifest_invalid"),
        (429, "function_rate_limited"),
        (429, "inspection_budget_exhausted"),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_api_errors_message_details_retry_after(sdk, status, code, nested):
    request = httpx.Request(
        "GET",
        "https://example.test/v1/sandbox/functions",
        headers={"Authorization": "Bearer unit-cloud-function-key"},
    )
    body = {
        "code": code,
        "message": "Server explains failure",
        "details": {"limit": 200, "reason": ["try later", "unit-cloud-function-key"]},
    }
    headers = {"Retry-After": "12"} if status == 429 else {}
    sdk.functions.list.side_effect = APIStatusError(
        "error",
        response=httpx.Response(status, request=request, headers=headers),
        body={"error": body} if nested else body,
    )
    result = invoke("list", "-j")
    assert result.exit_code == 1, result.output
    document = json.loads(result.stdout)
    assert document["error"] == "Server explains failure"
    assert document["code"] == code
    assert document["status"] == status
    assert document["details"] == {"limit": 200, "reason": ["try later", "[redacted]"]}
    assert "Server explains failure" in result.stderr
    assert "Details:" in result.stderr
    assert "unit-cloud-function-key" not in result.output
    if status == 429:
        assert document["retryAfter"] == "12"
        assert "Retry-After: 12" in result.stderr
    sdk.client.close.assert_called_once()


@pytest.mark.parametrize("source", ["env", "login"])
def test_client_reuses_credentials_and_base_url(sdk, monkeypatch, source):
    monkeypatch.setattr(
        selection,
        "resolve_cloud_credential",
        lambda: selection.CloudCredential("test-credential", source, source == "login"),
    )
    monkeypatch.setenv("MOBILERUN_API_BASE_URL", "https://dev.example.test/v1")
    result = invoke("list", "-j")
    assert result.exit_code == 0, result.output
    sdk.factory.assert_called_once_with(
        "test-credential", "https://dev.example.test/v1"
    )


def test_missing_credential_never_builds_client(sdk, monkeypatch):
    monkeypatch.delenv("MOBILERUN_API_KEY")
    result = invoke("list", "-j")
    assert result.exit_code == 1, result.output
    assert "MOBILERUN_API_KEY" in json.loads(result.stdout)["error"]
    sdk.factory.assert_not_called()


@pytest.mark.parametrize("force", [False, True])
def test_pull_failed_write_is_atomic(sdk, monkeypatch, tmp_path, force):
    output = tmp_path / "out.py"
    if force:
        output.write_bytes(b"original")
    make_temporary = function.tempfile.NamedTemporaryFile

    def temporary_file(**kwargs):
        stream = make_temporary(**kwargs)
        write = stream.write

        def fail(data):
            write(data[:3])
            raise OSError("disk write failed")

        stream.write = fail
        return stream

    monkeypatch.setattr(function.tempfile, "NamedTemporaryFile", temporary_file)
    result = invoke("pull", ID2, "-o", output, *(["--force"] if force else []), "-j")
    assert result.exit_code == 1, result.output
    assert "disk write failed" in json.loads(result.stdout)["error"]
    if force:
        assert output.read_bytes() == b"original"
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".mobilerun-pull-*"))


def test_pull_force_replaces_symlink(sdk, tmp_path):
    target = tmp_path / "target.py"
    target.write_bytes(b"keep target")
    output = tmp_path / "out.py"
    output.symlink_to(target)
    result = invoke("pull", ID2, "-o", output, "--force", "-j")
    assert result.exit_code == 0, result.output
    assert not output.is_symlink()
    assert output.read_bytes() == SOURCE.encode()
    assert target.read_bytes() == b"keep target"
    assert not list(tmp_path.glob(".mobilerun-pull-*"))


def test_pull_destination_race_preserves_existing_file(sdk, monkeypatch, tmp_path):
    output = tmp_path / "out.py"
    link = function.os.link

    def destination_appears(source, destination):
        output.write_bytes(b"raced")
        link(source, destination)

    monkeypatch.setattr(function.os, "link", destination_appears)
    result = invoke("pull", ID2, "-o", output, "-j")
    assert result.exit_code == 1, result.output
    assert "--force" in json.loads(result.stdout)["error"]
    assert output.read_bytes() == b"raced"
    assert not list(tmp_path.glob(".mobilerun-pull-*"))


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_pull_publish_failure_keeps_original(sdk, monkeypatch, tmp_path, operation):
    output = tmp_path / "out.py"
    output.write_bytes(b"original")
    monkeypatch.setattr(
        function.os, operation, Mock(side_effect=OSError("publish failed"))
    )
    result = invoke("pull", ID2, "-o", output, "--force", "-j")
    assert result.exit_code == 1, result.output
    assert "publish failed" in json.loads(result.stdout)["error"]
    assert output.read_bytes() == b"original"
    assert not list(tmp_path.glob(".mobilerun-pull-*"))


@pytest.mark.parametrize("shape", ["list", "dict", "string"])
def test_error_details_text_is_bounded_json_is_complete(sdk, shape):
    entries = [{"message": f"item {index}: " + "x" * 500} for index in range(15)]
    details = (
        entries
        if shape == "list"
        else (
            {str(index): entry for index, entry in enumerate(entries)}
            if shape == "dict"
            else "x" * 500
        )
    )
    request = httpx.Request("GET", "https://example.test/v1/sandbox/functions")
    sdk.functions.list.side_effect = APIStatusError(
        "invalid",
        response=httpx.Response(422, request=request),
        body={"message": "invalid", "details": details},
    )
    result = invoke("list", "-j")
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["details"] == details
    lines = (
        result.stderr.split("Details:\n", 1)[1]
        .split("Check the input fields", 1)[0]
        .splitlines()
    )
    assert all(len(line) <= 200 for line in lines)
    if shape != "string":
        assert len(lines) == 11
        assert lines[-1] == "… 5 more"
        assert "item 10" not in result.stderr
    else:
        assert len(lines) == 1
    assert lines[0].endswith("…")


ISSUES = [
    {
        "code": "no_entry",
        "message": "Entry point missing",
        "line": 4,
        "repair": {"action": "edit", "hint": "Define main"},
    },
    {"code": "no_schema", "message": "Schema missing"},
]
WARNINGS = [{"code": "slow", "message": "Slow import", "line": 1}]


def check(*args):
    return invoke("check", "--remote", *args)


def test_check_remote_valid_summary(sdk, file):
    sdk.functions.validations.create.return_value = SimpleNamespace(data=validation())
    result = check(file)
    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "Output kind: value\nDataset key: None\n"
        'Input schema:\n{\n  "type": "object"\n}\n'
        'Output schema:\n{\n  "type": "string"\n}\n'
    )
    assert result.stderr == ""
    sdk.functions.validations.create.assert_called_once_with(source=SOURCE)


def test_check_remote_polls_until_done(sdk, file, clock):
    sdk.functions.validations.create.return_value = SimpleNamespace(
        data=validation(status="pending", valid=None)
    )
    sdk.functions.validations.retrieve.side_effect = [
        SimpleNamespace(data=validation(status="pending", valid=None)),
        SimpleNamespace(data=validation()),
    ]
    result = check(file, "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "done"
    assert clock.sleeps == [0, 1]
    assert sdk.functions.validations.retrieve.call_count == 2
    assert sdk.functions.validations.retrieve.call_args.args == (ID1,)
    assert sdk.functions.validations.retrieve.call_args.kwargs["wait_s"] == 5


def test_check_remote_invalid_renders_issues(sdk, file):
    sdk.functions.validations.create.return_value = SimpleNamespace(
        data=validation(valid=False, errors=ISSUES, warnings=WARNINGS)
    )
    result = check(file)
    assert result.exit_code == 3, result.output
    assert result.stdout == ""
    assert result.stderr == (
        "line 4: Entry point missing — Define main\n"
        "Schema missing\nwarning: line 1: Slow import\n"
    )


def test_check_remote_invalid_json_is_server_document(sdk, file):
    sdk.functions.validations.create.return_value = SimpleNamespace(
        data=validation(valid=False, errors=ISSUES)
    )
    result = check(file, "-j")
    assert result.exit_code == 3, result.output
    document = json.loads(result.stdout)
    assert document["valid"] is False
    assert document["errors"][0]["line"] == 4
    assert result.stderr.startswith("line 4: Entry point missing")


def test_check_remote_failed_validation(sdk, file):
    sdk.functions.validations.create.return_value = SimpleNamespace(
        data=validation(status="failed", valid=None)
    )
    result = check(file)
    assert result.exit_code == 1, result.output
    assert result.stderr == "Validation failed\n"


def test_check_remote_pending_after_deadline(sdk, file, clock):
    pending = SimpleNamespace(data=validation(status="pending", valid=None))
    sdk.functions.validations.create.return_value = pending

    def long_poll(*args, **kwargs):
        clock.now += 5
        return pending

    sdk.functions.validations.retrieve.side_effect = long_poll
    result = check(file)
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{ID1}\n"
    assert "still pending" in result.stderr
    assert clock.sleeps[:3] == [0, 1, 1]
    assert sdk.functions.validations.retrieve.call_count == 10


def test_check_remote_no_wait(sdk, file, clock):
    sdk.functions.validations.create.return_value = SimpleNamespace(
        data=validation(status="pending", valid=None)
    )
    result = check(file, "--no-wait")
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{ID1}\n"
    assert f"mobilerun function check {file} --remote" in result.stderr
    assert clock.sleeps == []
    sdk.functions.validations.retrieve.assert_not_called()


def test_script_group_is_an_unknown_command(sdk):
    result = CliRunner().invoke(cli, ["script", "list", "-j"])
    assert result.exit_code == 2
    assert "No such command" in result.output
    assert not sdk.client.mock_calls


def test_push_creates_and_preserves_source(sdk, file):
    sdk.functions.list.return_value = page()
    result = invoke("push", file, "-j")
    assert result.exit_code == 0, result.output
    sdk.functions.create.assert_called_once_with(name="hello.py", source=SOURCE)
    assert json.loads(result.stdout)["latestVersion"]["version"] == 2


def test_push_existing_name_appends_version(sdk, file):
    result = invoke("push", file, "-j")
    assert result.exit_code == 0, result.output
    sdk.functions.versions.create.assert_called_once_with(ID2, source=SOURCE)
    sdk.functions.create.assert_not_called()


@pytest.mark.parametrize("status,expected", [("ready", 0), ("rejected", 1)])
def test_push_waits_for_inspection(sdk, file, clock, status, expected):
    sdk.functions.list.return_value = page()
    sdk.functions.create.return_value.data = record(status="pending_inspection")
    sdk.functions.retrieve.return_value.data = record(status=status)
    result = invoke("push", file, "-j")
    assert result.exit_code == expected, result.output
    assert clock.sleeps == [1]
    assert json.loads(result.stdout)["latestVersion"]["status"] == status


def test_function_list_pagination_and_exact_name(sdk):
    sdk.functions.list.side_effect = [
        page(record(name="prefix"), has_next=True),
        page(record()),
    ]
    result = invoke("list", "--name", "hello.py", "-j")
    assert result.exit_code == 0, result.output
    assert [item["name"] for item in json.loads(result.stdout)] == ["hello.py"]
    assert sdk.functions.list.call_args_list == [
        call(page=1, search="hello.py"),
        call(page=2, search="hello.py"),
    ]


def test_show_selected_version(sdk):
    from mobilerun_sdk.types.compute.functions.version_retrieve_response import Data

    sdk.functions.versions.retrieve.return_value.data = Data(
        **record().latest_version.model_dump(by_alias=True)
    )
    result = invoke("show", ID2, "--version", "2", "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["functionId"] == ID2
    sdk.functions.versions.retrieve.assert_called_once_with(2, function_id=ID2)


def test_versions_uses_version_resource(sdk):
    from mobilerun_sdk.types.compute.functions.version_list_response import (
        VersionListResponse,
    )

    sdk.functions.versions.list.return_value = VersionListResponse(
        data={
            "items": [record().latest_version.model_dump(by_alias=True)],
            "pagination": {
                "hasNext": False,
                "hasPrev": False,
                "page": 1,
                "pageSize": 20,
                "total": 1,
                "pages": 1,
            },
        }
    )
    result = invoke("versions", ID2, "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)[0]["version"] == 2
    sdk.functions.versions.list.assert_called_once_with(ID2, page=1)


@pytest.mark.parametrize("version", [None, 2])
def test_pull_version_and_integrity(sdk, tmp_path, version):
    target = tmp_path / "download.py"
    result = invoke(
        "pull", ID2, "-o", target, *(["--version", version] if version else [])
    )
    assert result.exit_code == 0, result.output
    assert target.read_bytes() == SOURCE.encode()
    sdk.functions.source.assert_called_once_with(
        ID2, **({"version": 2} if version else {})
    )


def test_pull_rejects_hash_mismatch(sdk, tmp_path):
    sdk.functions.source.return_value.data.sha256 = "bad"
    target = tmp_path / "download.py"
    result = invoke("pull", ID2, "-o", target)
    assert result.exit_code == 1
    assert "SHA-256" in result.stderr
    assert not target.exists()


@pytest.mark.parametrize("version", [None, 2])
def test_delete_resource_or_version(sdk, version):
    result = invoke(
        "delete", ID2, "--yes", *(["--version", version] if version else []), "-j"
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["deleted"] is True
    if version:
        sdk.functions.versions.delete.assert_called_once_with(2, function_id=ID2)
        sdk.functions.delete.assert_not_called()
    else:
        sdk.functions.delete.assert_called_once_with(ID2)


def run_record(status="done", **updates):
    from mobilerun_sdk.types.compute.run_retrieve_response import Data

    return Data(
        **{
            **dict(
                id=ID1,
                functionId=ID2,
                version=2,
                input={},
                logs="pending",
                status=status,
                stdout="",
                stderr="",
                truncated=False,
                result={"value": 7},
            ),
            **updates,
        }
    )


def test_cloud_run_wait_and_input(sdk):
    sdk.functions.runs.create.return_value.data = run_record("running")
    sdk.client.compute.runs.retrieve.return_value.data = run_record(logs="available")
    from mobilerun_sdk.types.compute.run_logs_response import RunLogsResponse

    sdk.client.compute.runs.logs.return_value = RunLogsResponse(
        data={
            "url": "https://example.test/logs",
            "sizeBytes": 42,
            "expiresAt": "2026-11-24T12:00:00Z",
        }
    )
    result = invoke(
        "run", ID2, "--input", '{"name":"Ada"}', "--version", "2", "--wait", "-j"
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["result"] == {"value": 7}
    assert json.loads(result.stdout)["logs"] == "available"
    sdk.functions.runs.create.assert_called_once_with(
        ID2, input={"name": "Ada"}, wait_s=0, version=2
    )
    assert sdk.client.compute.runs.retrieve.call_count == 2
    sdk.client.compute.runs.logs.assert_called_once()
    assert (
        json.loads(result.stdout)["logDownload"]["url"] == "https://example.test/logs"
    )


def test_cloud_run_failure_is_nonzero(sdk):
    sdk.functions.runs.create.return_value.data = run_record(
        "failed", errorCode="input_invalid"
    )
    result = invoke("run", ID2, "--input", "{}")
    assert result.exit_code == 1
    assert "input_invalid" in result.stderr


@pytest.mark.parametrize(
    "args", [["--input", "[]"], ["--input", "{}", "--device", "phone"]]
)
def test_cloud_run_rejects_unsupported_input(sdk, args):
    result = invoke("run", ID2, *args)
    assert result.exit_code == 2
    sdk.functions.runs.create.assert_not_called()


def test_function_logs_uses_sdk_wait_logs_before_logs(sdk, monkeypatch):
    from mobilerun_sdk import Run
    from mobilerun_sdk.types.compute.run_logs_response import RunLogsResponse

    wait = Mock(return_value=SimpleNamespace(data=run_record(logs="available")))
    logs = Mock(
        return_value=RunLogsResponse(
            data={
                "url": "https://example.test/logs",
                "sizeBytes": 42,
                "expiresAt": "2026-11-24T12:00:00Z",
            }
        )
    )
    calls = Mock()
    calls.attach_mock(wait, "wait_logs")
    calls.attach_mock(logs, "logs")
    monkeypatch.setattr(Run, "wait_logs", wait)
    monkeypatch.setattr(Run, "logs", logs)
    result = invoke("logs", ID1, "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["url"] == "https://example.test/logs"
    assert calls.mock_calls == [
        call.wait_logs(timeout=60, poll_interval=1),
        call.logs(),
    ]


def test_function_logs_deadline_has_clear_error(sdk, monkeypatch):
    from mobilerun_sdk import Run, WaitTimeoutError

    monkeypatch.setattr(
        Run, "wait_logs", Mock(side_effect=WaitTimeoutError("Wait timed out"))
    )
    result = invoke("logs", ID1, "-j")
    assert result.exit_code == 1
    assert "retry `mobilerun function logs" in result.stderr
    sdk.client.compute.runs.logs.assert_not_called()


def test_cloud_run_deadline_preserves_identity(sdk, clock):
    sdk.functions.runs.create.return_value.data = run_record("running")
    sdk.client.compute.runs.retrieve.return_value.data = run_record("running")
    result = invoke("run", ID2, "--input", "{}", "--wait", "-j")
    assert result.exit_code == 124
    assert json.loads(result.stdout)["id"] == ID1
    assert "still running" in result.stderr
    assert sum(clock.sleeps) == 60
    sdk.client.compute.runs.logs.assert_not_called()


def test_logs_gone_explains_and_never_requests_url(sdk, monkeypatch):
    from mobilerun_sdk import Run, WaitFailureError

    monkeypatch.setattr(
        Run, "wait_logs", Mock(side_effect=WaitFailureError(run_record(logs="gone")))
    )
    result = invoke("logs", ID1, "-j")
    assert result.exit_code == 1
    assert "unavailable or expired" in result.stderr
    sdk.client.compute.runs.logs.assert_not_called()


@pytest.mark.parametrize(
    "command,method",
    [("versions", "versions.list"), ("show", "retrieve"), ("delete", "delete")],
)
def test_resource_failures_have_nonzero_exit(sdk, command, method):
    from operator import attrgetter

    failure = APIStatusError(
        "missing",
        response=httpx.Response(
            404,
            request=httpx.Request(
                "GET", "https://example.test/v1/compute/functions/missing"
            ),
        ),
        body={"message": "Function not found"},
    )
    attrgetter(method)(sdk.functions).side_effect = failure
    result = invoke(command, ID2, *(["--yes"] if command == "delete" else []), "-j")
    assert result.exit_code == 1
    assert "not found" in result.stderr


def test_delete_interactive_cancel_never_mutates(sdk, monkeypatch):
    from mobilerun.cli import options

    monkeypatch.setattr(options, "input_allowed", lambda value: True)
    result = invoke("delete", ID2, input="n\n")
    assert result.exit_code == 1
    assert "cancelled" in result.stderr
    sdk.functions.delete.assert_not_called()
    sdk.functions.versions.delete.assert_not_called()


def test_cloud_run_ignores_inherited_device_settings(sdk, monkeypatch):
    monkeypatch.setenv("MOBILERUN_TARGET", "cloud")
    monkeypatch.setenv("MOBILERUN_DEVICE", "inherited-phone")
    sdk.functions.runs.create.return_value.data = run_record()
    result = invoke("run", ID2, "--input", "{}", "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["id"] == ID1
    sdk.functions.runs.create.assert_called_once_with(ID2, input={}, wait_s=0)


@pytest.mark.parametrize(
    "flags", [["--target", "cloud"], ["--target", "local"], ["--device", "phone"]]
)
def test_cloud_run_rejects_only_explicit_device_flags(sdk, monkeypatch, flags):
    monkeypatch.setenv("MOBILERUN_TARGET", "cloud")
    monkeypatch.setenv("MOBILERUN_DEVICE", "inherited-phone")
    result = invoke("run", ID2, "--input", "{}", *flags)
    assert result.exit_code == 2
    assert "--local" in result.stderr
    sdk.factory.assert_not_called()


@pytest.mark.parametrize("args", [[], ["--version", "2", "--all"]])
def test_delete_name_requires_exactly_one_version_choice(sdk, args):
    result = invoke("delete", "hello.py", "--yes", *args)
    assert result.exit_code == 2
    assert "--version" in result.stderr
    assert "--all" in result.stderr
    sdk.functions.delete.assert_not_called()
    sdk.functions.versions.delete.assert_not_called()


def test_delete_name_all_deletes_function(sdk):
    result = invoke("delete", "hello.py", "--all", "--yes", "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["deleted"] is True
    sdk.functions.delete.assert_called_once_with(ID2)


def test_delete_name_version_reports_selected_version(sdk):
    result = invoke("delete", "hello.py", "--version", "2", "--yes")
    assert result.exit_code == 0, result.output
    assert result.stdout == "Deleted hello.py version 2\n"
    sdk.functions.versions.delete.assert_called_once_with(2, function_id=ID2)
    sdk.functions.delete.assert_not_called()


@pytest.mark.parametrize(
    "status,outcome,expected", [("done", "succeeded", 0), ("failed", "failed", 1)]
)
def test_finished_run_without_logs_keeps_its_exit_code(sdk, status, outcome, expected):
    sdk.functions.runs.create.return_value.data = run_record("running")
    sdk.client.compute.runs.retrieve.return_value.data = run_record(
        status, logs="none", outcome=outcome
    )
    result = invoke("run", ID2, "--input", "{}", "--wait", "-j")
    assert result.exit_code == expected, result.output
    assert json.loads(result.stdout)["status"] == status
    assert json.loads(result.stdout)["logs"] == "none"
    assert "warning: Could not retrieve logs" in result.stderr
    sdk.client.compute.runs.logs.assert_not_called()


def test_finished_run_with_slow_logs_warns_and_succeeds(sdk, monkeypatch):
    from mobilerun_sdk import Run, WaitTimeoutError

    sdk.functions.runs.create.return_value.data = run_record("running")
    sdk.client.compute.runs.retrieve.return_value.data = run_record(outcome="succeeded")
    monkeypatch.setattr(
        Run, "wait_logs", Mock(side_effect=WaitTimeoutError("Wait timed out"))
    )
    result = invoke("run", ID2, "--input", "{}", "--wait", "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["outcome"] == "succeeded"
    assert "warning: Could not retrieve logs" in result.stderr


def test_cloud_run_custom_timeout_spans_deadline(sdk, clock):
    sdk.functions.runs.create.return_value.data = run_record("running")
    sdk.client.compute.runs.retrieve.return_value.data = run_record("running")
    result = invoke("run", ID2, "--input", "{}", "--wait", "--timeout", "3", "-j")
    assert result.exit_code == 124
    assert json.loads(result.stdout)["id"] == ID1
    assert "still running" in result.stderr
    assert sum(clock.sleeps) == 3


def test_versions_tsv_uses_iso_timestamp(sdk):
    from mobilerun_sdk.types.compute.functions.version_list_response import (
        VersionListResponse,
    )

    sdk.functions.versions.list.return_value = VersionListResponse(
        data={
            "items": [record().latest_version.model_dump(by_alias=True)],
            "pagination": {
                "hasNext": False,
                "hasPrev": False,
                "page": 1,
                "pageSize": 20,
                "total": 1,
                "pages": 1,
            },
        }
    )
    result = invoke("versions", ID2)
    assert result.exit_code == 0, result.output
    assert result.stdout == "2\tready\t2026-11-13T00:00:00+00:00\n"


def test_show_tsv_uses_iso_timestamps_in_metadata(sdk):
    result = invoke("show", ID2)
    assert result.exit_code == 0, result.output
    rows = dict(line.split("\t", 1) for line in result.stdout.splitlines())
    assert rows["createdAt"] == "2026-11-13T00:00:00+00:00"
    assert json.loads(rows["latestVersion"])["createdAt"] == "2026-11-13T00:00:00+00:00"


def test_best_effort_log_warning_redacts_api_credentials(sdk, monkeypatch):
    from mobilerun_sdk import Run

    sdk.functions.runs.create.return_value.data = run_record("running")
    sdk.client.compute.runs.retrieve.return_value.data = run_record(outcome="succeeded")
    error = APIStatusError(
        "error",
        response=httpx.Response(
            403, request=httpx.Request("GET", "https://example.test/logs")
        ),
        body={"message": "Rejected unit-cloud-function-key"},
    )
    monkeypatch.setattr(Run, "wait_logs", Mock(side_effect=error))
    result = invoke("run", ID2, "--input", "{}", "--wait", "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["outcome"] == "succeeded"
    assert "unit-cloud-function-key" not in result.output
    assert "warning: Could not retrieve logs" in result.stderr
    assert "[redacted]" in result.stderr


def test_cloud_run_execution_timeout_returns_124(sdk):
    sdk.functions.runs.create.return_value.data = run_record(
        "failed", errorCode="timeout"
    )
    result = invoke("run", ID2, "--input", "{}", "-j")
    assert result.exit_code == 124
    assert json.loads(result.stdout)["errorCode"] == "timeout"


def test_function_run_help_documents_timeout_exit():
    result = invoke("run", "--help")
    assert result.exit_code == 0
    assert "124" in result.stdout


def test_list_rejects_name_with_search(sdk):
    result = invoke("list", "--name", "hello.py", "--search", "hel")
    assert result.exit_code == 2
    assert "--name or --search" in result.stderr
    sdk.factory.assert_not_called()
