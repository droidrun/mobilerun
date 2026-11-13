"""Cloud script contracts against a fake generated SDK client, without network."""

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
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    NotFoundError,
)
from mobilerun_sdk.types.script import Script
from mobilerun_sdk.types.script_source import ScriptSource
from mobilerun_sdk.types.script_source_response import ScriptSourceResponse

from mobilerun.cli import deviceauth, script, selection
from mobilerun.cli.main import cli
from mobilerun.config_manager import ConfigLoader

ID1 = "31fe8bac-cf2a-44af-af62-88aa6c73c0a1"
ID2 = "31fe8bac-cf2a-44af-af62-88aa6c73c0a2"
SOURCE = '# café\r\nprint("hello")\r\n'
DIGEST = hashlib.sha256(SOURCE.encode()).hexdigest()


def record(**updates):
    values = dict(
        id=ID2,
        name="hello.py",
        version=2,
        versionCount=2,
        status="ready",
        createdAt=datetime(2026, 11, 13, tzinfo=timezone.utc),
        createdBy="test-owner",
        manifest={"hosts": ["example.com"], "credentials": [], "scopes": []},
        inputSchema={"type": "object"},
        outputSchema={"type": "string"},
        sha256=DIGEST,
        sizeBytes=len(SOURCE.encode()),
    )
    values.update(updates)
    return Script(**values)


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
    monkeypatch.setenv("MOBILERUN_API_KEY", "unit-cloud-script-key")
    monkeypatch.setattr(deviceauth, "load_credentials", lambda: None)
    monkeypatch.setattr(script, "input_allowed", lambda no_input: False)
    client = Mock()
    client.with_options.return_value = client
    factory = Mock(return_value=client)
    monkeypatch.setattr(cloud, "make_sdk_client", factory)
    client.scripts.create.return_value = SimpleNamespace(data=record())
    client.scripts.retrieve.return_value = SimpleNamespace(data=record())
    client.scripts.list.return_value = page(record(), record(id=ID1, version=1))
    client.scripts.source.return_value = ScriptSourceResponse(
        data=ScriptSource(
            scriptId=ID2, sha256=DIGEST, sizeBytes=len(SOURCE.encode()), source=SOURCE
        )
    )
    return SimpleNamespace(client=client, scripts=client.scripts, factory=factory)


@pytest.fixture
def file(tmp_path):
    path = tmp_path / "hello.py"
    path.write_bytes(SOURCE.encode())
    return path


def invoke(*args, **kwargs):
    return CliRunner().invoke(cli, ["script", *map(str, args)], **kwargs)


def test_push_default_name_and_exact_source(sdk, file):
    first = invoke("push", file, "-j")
    second = invoke("push", file, "-j")
    assert first.exit_code == second.exit_code == 0, first.output + second.output
    calls = sdk.scripts.create.call_args_list
    assert all(entry.kwargs["name"] == "hello.py" for entry in calls)
    assert all(entry.kwargs["source"].encode() == file.read_bytes() for entry in calls)
    assert "manifest" not in calls[0].kwargs
    assert "description" not in calls[0].kwargs
    assert json.loads(first.stdout)["version"] == 2
    assert first.stderr == ""
    assert sdk.client.close.call_count == 2


def test_push_manifest_and_explicit_name(sdk, file, tmp_path):
    manifest = {
        "hosts": ["example.com"],
        "credentials": [{"name": "login", "packageName": "com.example"}],
        "scopes": ["device.read"],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    result = invoke(
        "push",
        file,
        "--name",
        "named",
        "--description",
        "description",
        "--manifest",
        path,
    )
    assert result.exit_code == 0, result.output
    params = sdk.scripts.create.call_args.kwargs
    assert params["name"] == "named"
    assert params["description"] == "description"
    assert params["manifest"] == manifest
    assert result.stdout == f"hello.py\t2\t{ID2}\tready\n"


@pytest.mark.parametrize("value", ["{", "[]"])
def test_push_invalid_manifest_before_network(sdk, file, tmp_path, value):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(value)
    result = invoke("push", file, "--manifest", manifest, "-j")
    assert result.exit_code == 2
    assert "--manifest" in result.stderr
    sdk.factory.assert_not_called()


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=0, sleeps=[])

    def sleep(seconds):
        state.sleeps.append(seconds)
        state.now += seconds

    monkeypatch.setattr(script.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(script.time, "sleep", sleep)
    return state


@pytest.mark.parametrize("status,exit_code", [("ready", 0), ("rejected", 1)])
@pytest.mark.parametrize("json_output", [False, True])
def test_push_wait_inspection(sdk, file, clock, status, exit_code, json_output):
    sdk.scripts.create.return_value.data = record(status="pending_inspection")
    error = (
        {"code": "invalid_entrypoint", "message": "entrypoint missing"}
        if status == "rejected"
        else None
    )
    sdk.scripts.retrieve.side_effect = [
        SimpleNamespace(data=record(status="pending_inspection")),
        SimpleNamespace(data=record(status=status, inspectError=error)),
    ]
    result = invoke("push", file, *(["-j"] if json_output else []))
    assert result.exit_code == exit_code, result.output
    assert clock.sleeps == [1, 2]
    assert sdk.scripts.retrieve.call_count == 2
    if json_output:
        document = json.loads(result.stdout)
        assert document["status"] == status
        assert document["inspectError"] == error
    else:
        assert result.stdout == f"hello.py\t2\t{ID2}\t{status}\n"
    if error:
        assert "entrypoint missing" in result.stderr


def test_push_wait_timeout_is_success(sdk, file, clock):
    pending = record(status="pending_inspection")
    sdk.scripts.create.return_value.data = pending
    sdk.scripts.retrieve.return_value.data = pending
    result = invoke("push", file, "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "pending_inspection"
    assert "mobilerun script show hello.py" in result.stderr
    assert sum(clock.sleeps) == 60
    assert clock.sleeps[:3] == [1, 2, 2]


def test_push_no_wait(sdk, file, clock):
    sdk.scripts.create.return_value.data = record(status="pending_inspection")
    result = invoke("push", file, "--no-wait", "-j")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "pending_inspection"
    assert clock.sleeps == []
    sdk.scripts.retrieve.assert_not_called()
    assert result.stderr == ""


@pytest.mark.parametrize("by_name", [False, True])
def test_list_pages_and_modes(sdk, by_name):
    sdk.scripts.list.side_effect = [
        page(record(id=ID1, version=1), has_next=True),
        page(record()),
    ]
    result = invoke(
        "list", "--search", "hello", *(["--name", "hello.py"] if by_name else []), "-j"
    )
    assert result.exit_code == 0, result.output
    documents = json.loads(result.stdout)
    assert [item["version"] for item in documents] == ([2, 1] if by_name else [1, 2])
    filters = {"name": "hello.py"} if by_name else {"latest": "true"}
    assert sdk.scripts.list.call_args_list == [
        call(page=1, search="hello", **filters),
        call(page=2, search="hello", **filters),
    ]


def test_list_text_version_count(sdk):
    sdk.scripts.list.return_value = page(record())
    result = invoke("list")
    assert result.exit_code == 0, result.output
    assert result.stdout == "hello.py\t2\t2\tready\t2026-11-13T00:00:00+00:00\n"


@pytest.mark.parametrize(
    "argument,version,expected",
    [(ID2, None, 2), ("hello.py", None, 2), ("hello.py", 1, 1)],
)
def test_show_resolution_and_metadata(sdk, argument, version, expected):
    result = invoke(
        "show", argument, *(["--version", version] if version else []), "-j"
    )
    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert document["version"] == expected
    assert document["manifest"]["hosts"] == ["example.com"]
    assert document["inputSchema"] == {"type": "object"}
    if argument == ID2:
        sdk.scripts.retrieve.assert_called_once_with(ID2)
        sdk.scripts.list.assert_not_called()
    else:
        sdk.scripts.list.assert_called_once_with(page=1, name="hello.py")


def test_show_latest_even_if_server_order_differs(sdk):
    sdk.scripts.list.return_value = page(record(version=1), record(version=3))
    result = invoke("show", "hello.py")
    assert result.exit_code == 0, result.output
    assert "version\t3\n" in result.stdout
    rows = dict(line.split("\t", 1) for line in result.stdout.splitlines())
    assert json.loads(rows["manifest"])["hosts"] == ["example.com"]
    assert "inspectError\t\n" in result.stdout


@pytest.mark.parametrize("command", ["show", "pull", "delete"])
@pytest.mark.parametrize("missing_name", [True, False])
def test_resolution_errors(sdk, command, missing_name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    if missing_name:
        sdk.scripts.list.return_value = page()
    result = invoke(
        command,
        "missing",
        "--version",
        9,
        *(["--yes"] if command == "delete" else []),
        "-j",
    )
    assert result.exit_code == 1, result.output
    message = "No script named missing" if missing_name else "Available versions: 2, 1"
    assert message in json.loads(result.stdout)["error"]
    assert message in result.stderr
    sdk.scripts.delete.assert_not_called()
    sdk.scripts.source.assert_not_called()


@pytest.mark.parametrize(
    "argument,expected",
    [("hello.py", "hello.py"), ("hello", "hello.py"), (ID2, ID2 + ".py")],
)
def test_pull_uses_argument_not_server_name(
    sdk, monkeypatch, tmp_path, argument, expected
):
    monkeypatch.chdir(tmp_path)
    sdk.scripts.retrieve.return_value.data = record(name="../../untrusted.py")
    sdk.scripts.list.return_value = page(record(name="../../untrusted.py"))
    result = invoke("pull", argument, "-j")
    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert document == {
        "id": ID2,
        "name": "../../untrusted.py",
        "version": 2,
        "path": expected,
        "sha256": DIGEST,
        "sizeBytes": len(SOURCE.encode()),
    }
    assert (tmp_path / expected).read_bytes() == SOURCE.encode()
    sdk.scripts.source.assert_called_once_with(ID2)


@pytest.mark.parametrize("force", [False, True])
def test_pull_overwrite_and_version(sdk, tmp_path, force):
    output = tmp_path / "out.py"
    output.write_bytes(b"original")
    result = invoke(
        "pull",
        "hello.py",
        "--version",
        1,
        "-o",
        output,
        *(["--force"] if force else []),
    )
    assert result.exit_code == (0 if force else 1), result.output
    assert output.read_bytes() == (SOURCE.encode() if force else b"original")
    if force:
        sdk.scripts.source.assert_called_once_with(ID1)
        assert result.stdout == str(output) + "\n"
    else:
        assert "--force" in result.stderr
        sdk.factory.assert_not_called()


@pytest.mark.parametrize("exists", [False, True])
def test_pull_hash_mismatch_never_writes(sdk, tmp_path, exists):
    output = tmp_path / "out.py"
    if exists:
        output.write_bytes(b"keep")
    sdk.scripts.source.return_value.data.sha256 = "0" * 64
    result = invoke("pull", ID2, "-o", output, "--force", "-j")
    assert result.exit_code == 1, result.output
    assert "SHA-256" in json.loads(result.stdout)["error"]
    assert output.read_bytes() == b"keep" if exists else not output.exists()


@pytest.mark.parametrize("all_versions", [False, True])
def test_delete_selected_versions(sdk, all_versions):
    result = invoke(
        "delete",
        "hello.py",
        *(["--all"] if all_versions else ["--version", 1]),
        "--yes",
        "-j",
    )
    assert result.exit_code == 0, result.output
    deleted = json.loads(result.stdout)["deleted"]
    assert [item["version"] for item in deleted] == ([2, 1] if all_versions else [1])
    assert sdk.scripts.delete.call_args_list == (
        [call(ID2), call(ID1)] if all_versions else [call(ID1)]
    )


def test_delete_id(sdk):
    result = invoke("delete", ID2, "--yes")
    assert result.exit_code == 0, result.output
    sdk.scripts.delete.assert_called_once_with(ID2)
    assert result.stdout == f"hello.py\t2\t{ID2}\n"


@pytest.mark.parametrize("args", [["hello.py"], ["hello.py", "--version", 1, "--all"]])
def test_delete_usage(sdk, args):
    result = invoke("delete", *args, "--yes")
    assert result.exit_code == 2, result.output
    sdk.factory.assert_not_called()


@pytest.mark.parametrize("no_input", [False, True])
def test_delete_requires_yes_without_input(sdk, monkeypatch, no_input):
    monkeypatch.setattr(
        script, "input_allowed", lambda disabled: not disabled and no_input
    )
    result = invoke(
        "delete", "hello.py", "--all", *(["--no-input"] if no_input else []), "-j"
    )
    assert result.exit_code == 2, result.output
    assert "--yes" in result.stderr
    sdk.factory.assert_not_called()


@pytest.mark.parametrize("answer,expected", [("y\n", 0), ("n\n", 1)])
def test_delete_confirmation_once(sdk, monkeypatch, answer, expected):
    monkeypatch.setattr(script, "input_allowed", lambda disabled: not disabled)
    result = invoke("delete", "hello.py", "--all", "-j", input=answer)
    assert result.exit_code == expected, result.output
    assert result.stderr.count("Delete 2 version(s)") == 1
    if expected == 0:
        assert len(json.loads(result.stdout)["deleted"]) == 2
        assert sdk.scripts.delete.call_count == 2
    else:
        sdk.scripts.delete.assert_not_called()


@pytest.mark.parametrize(
    "status,code",
    [
        (409, "script_limit_reached"),
        (409, "script_row_limit_reached"),
        (409, "script_busy"),
        (422, "script_name_invalid"),
        (422, "manifest_invalid"),
        (429, "script_rate_limited"),
        (429, "inspection_budget_exhausted"),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_api_errors_message_details_retry_after(sdk, status, code, nested):
    request = httpx.Request(
        "GET",
        "https://example.test/v1/sandbox/scripts",
        headers={"Authorization": "Bearer unit-cloud-script-key"},
    )
    body = {
        "code": code,
        "message": "Server explains failure",
        "details": {"limit": 200, "reason": ["try later", "unit-cloud-script-key"]},
    }
    headers = {"Retry-After": "12"} if status == 429 else {}
    sdk.scripts.list.side_effect = APIStatusError(
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
    assert "unit-cloud-script-key" not in result.output
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


@pytest.mark.parametrize("expire", [False, True])
def test_push_poll_timeout_respects_wait_budget(sdk, file, clock, expire):
    sdk.scripts.create.return_value.data = record(status="pending_inspection")

    def request_timeout(*args, **kwargs):
        if expire:
            clock.now += kwargs["timeout"].read
        raise APITimeoutError(request=httpx.Request("GET", "https://example.test"))

    sdk.scripts.retrieve.side_effect = request_timeout
    result = invoke("push", file, "-j")
    assert result.exit_code == 0, result.output
    sdk.client.with_options.assert_called_once_with(max_retries=0)
    assert "still pending" in result.stderr
    assert json.loads(result.stdout)["status"] == "pending_inspection"
    timeout = sdk.scripts.retrieve.call_args.kwargs["timeout"]
    assert timeout == httpx.Timeout(10)


@pytest.mark.parametrize("kind", ["connection", "rate_limit", "server"])
@pytest.mark.parametrize("json_output", [False, True])
def test_push_poll_error_keeps_created_version(sdk, file, clock, kind, json_output):
    sdk.scripts.create.return_value.data = record(status="pending_inspection")
    request = httpx.Request("GET", "https://example.test/v1/sandbox/scripts/" + ID2)
    error = (
        APIConnectionError(request=request)
        if kind == "connection"
        else APIStatusError(
            "poll failed",
            response=httpx.Response(
                429 if kind == "rate_limit" else 503, request=request
            ),
            body={"message": "poll failed"},
        )
    )
    sdk.scripts.retrieve.side_effect = error
    result = invoke("push", file, *(["-j"] if json_output else []))
    assert result.exit_code == 0, result.output
    assert "mobilerun script show hello.py" in result.stderr
    if json_output:
        document = json.loads(result.stdout)
        assert document["id"] == ID2
        assert document["version"] == 2
        assert document["status"] == "pending_inspection"
    else:
        assert result.stdout == f"hello.py\t2\t{ID2}\tpending_inspection\n"
    sdk.scripts.retrieve.assert_called_once()


def test_push_poll_uses_remaining_budget(sdk, file, clock):
    sdk.scripts.create.return_value.data = record(status="pending_inspection")

    def retrieve(*args, **kwargs):
        assert kwargs["timeout"] == httpx.Timeout(10 if clock.now == 1 else 2)
        clock.now = 56 if clock.now == 1 else 60
        return SimpleNamespace(data=record(status="pending_inspection"))

    sdk.scripts.retrieve.side_effect = retrieve
    result = invoke("push", file, "-j")
    assert result.exit_code == 0, result.output
    assert sdk.scripts.retrieve.call_count == 2


@pytest.mark.parametrize("force", [False, True])
def test_pull_failed_write_is_atomic(sdk, monkeypatch, tmp_path, force):
    output = tmp_path / "out.py"
    if force:
        output.write_bytes(b"original")
    make_temporary = script.tempfile.NamedTemporaryFile

    def temporary_file(**kwargs):
        stream = make_temporary(**kwargs)
        write = stream.write

        def fail(data):
            write(data[:3])
            raise OSError("disk write failed")

        stream.write = fail
        return stream

    monkeypatch.setattr(script.tempfile, "NamedTemporaryFile", temporary_file)
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
    link = script.os.link

    def destination_appears(source, destination):
        output.write_bytes(b"raced")
        link(source, destination)

    monkeypatch.setattr(script.os, "link", destination_appears)
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
        script.os, operation, Mock(side_effect=OSError("publish failed"))
    )
    result = invoke("pull", ID2, "-o", output, "--force", "-j")
    assert result.exit_code == 1, result.output
    assert "publish failed" in json.loads(result.stdout)["error"]
    assert output.read_bytes() == b"original"
    assert not list(tmp_path.glob(".mobilerun-pull-*"))


def not_found():
    request = httpx.Request("GET", "https://example.test/v1/sandbox/scripts/" + ID1)
    return NotFoundError(
        "missing",
        response=httpx.Response(404, request=request),
        body={"message": "missing"},
    )


@pytest.mark.parametrize(
    "command,flags",
    [
        ("show", []),
        ("pull", []),
        ("delete", ["--version", 1, "--yes"]),
        ("delete", ["--all", "--yes"]),
    ],
)
def test_uuid_shaped_name_falls_back_after_not_found(
    sdk, monkeypatch, tmp_path, command, flags
):
    monkeypatch.chdir(tmp_path)
    sdk.scripts.retrieve.side_effect = not_found()
    sdk.scripts.list.return_value = page(record(name=ID1))
    if "--version" in flags:
        sdk.scripts.list.return_value = page(record(name=ID1, version=1))
    result = invoke(command, ID1, *flags, "-j")
    assert result.exit_code == 0, result.output
    sdk.scripts.retrieve.assert_called_once_with(ID1)
    sdk.scripts.list.assert_called_once_with(page=1, name=ID1)
    if command == "delete":
        sdk.scripts.delete.assert_called_once_with(ID2)
        assert json.loads(result.stdout)["deleted"][0]["name"] == ID1
    elif command == "pull":
        assert (tmp_path / (ID1 + ".py")).read_bytes() == SOURCE.encode()
    else:
        assert json.loads(result.stdout)["name"] == ID1


@pytest.mark.parametrize(
    "name", [ID1.replace("-", ""), "{" + ID1 + "}", "urn:uuid:" + ID1]
)
def test_noncanonical_uuid_is_a_name(sdk, name):
    result = invoke("show", name, "-j")
    assert result.exit_code == 0, result.output
    sdk.scripts.retrieve.assert_not_called()
    sdk.scripts.list.assert_called_once_with(page=1, name=name)


def test_delete_all_rejects_resolved_id(sdk):
    result = invoke("delete", ID2, "--all", "--yes")
    assert result.exit_code == 2, result.output
    assert "--all requires a script name" in result.stderr
    sdk.scripts.list.assert_not_called()
    sdk.scripts.delete.assert_not_called()


def test_delete_uuid_name_still_requires_version_or_all(sdk):
    sdk.scripts.retrieve.side_effect = not_found()
    result = invoke("delete", ID1, "--yes")
    assert result.exit_code == 2, result.output
    assert "--version N or --all" in result.stderr
    sdk.scripts.delete.assert_not_called()


def test_uuid_retrieve_forbidden_does_not_fall_back(sdk):
    request = httpx.Request("GET", "https://example.test/v1/sandbox/scripts/" + ID1)
    sdk.scripts.retrieve.side_effect = APIStatusError(
        "forbidden",
        response=httpx.Response(403, request=request),
        body={"message": "forbidden"},
    )
    result = invoke("show", ID1, "-j")
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["status"] == 403
    sdk.scripts.list.assert_not_called()


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("completed", [0, 1])
@pytest.mark.parametrize("kind", ["status", "connection", "other"])
def test_delete_partial_failure_reports_completed(sdk, json_output, completed, kind):
    request = httpx.Request("DELETE", "https://example.test/v1/sandbox/scripts/" + ID1)
    if kind == "status":
        error = APIStatusError(
            "delete failed",
            response=httpx.Response(409, request=request),
            body={
                "code": "script_busy",
                "message": "delete failed",
                "details": ["in use"],
            },
        )
    elif kind == "connection":
        error = APIConnectionError(request=request)
    else:
        error = RuntimeError("delete failed")
    sdk.scripts.list.return_value = page(
        record(), record(id=ID1, version=1), record(id="third", version=3)
    )
    sdk.scripts.delete.side_effect = [None] * completed + [error, None, None]
    result = invoke(
        "delete", "hello.py", "--all", "--yes", *(["-j"] if json_output else [])
    )
    assert result.exit_code == 1, result.output
    assert sdk.scripts.delete.call_count == completed + 1
    sdk.scripts.delete.assert_has_calls([call("third"), call(ID2)][: completed + 1])
    if json_output:
        document = json.loads(result.stdout)
        assert document["deleted"] == (
            [{"id": "third", "name": "hello.py", "version": 3}] if completed else []
        )
        if kind == "status":
            assert document["error"] == {
                "error": "delete failed",
                "status": 409,
                "code": "script_busy",
                "details": ["in use"],
            }
        elif kind == "connection":
            assert document["error"]["error"] == "Could not connect to the API"
        else:
            assert document["error"] == {"error": "delete failed"}
    else:
        assert result.stdout == ("hello.py\t3\tthird\n" if completed else "")
    assert (
        "Could not connect" if kind == "connection" else "delete failed"
    ) in result.stderr
    sdk.client.close.assert_called_once()
