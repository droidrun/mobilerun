"""Manage Serverless Functions and test them locally."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import click

from mobilerun.cli.options import (
    INVALID_EXIT_CODE,
    Group,
    api_url_option,
    cloud_client,
    device_option,
    emit,
    emit_rows,
    examples,
    input_allowed,
    json_option,
    no_input_option,
    platform_option,
    print_issues,
    target_option,
    verbose_option,
    write_file,
)

# Total wait for inspection or validation, and the server-side long-poll per request.
WAIT_SECONDS = 60
VALIDATION_POLL_SECONDS = 5

python_option = click.option(
    "--python", "python_path", help="Python interpreter for the function."
)


def _interpreter(python_path):
    if python_path:
        interpreter = python_path
    elif os.environ.get("VIRTUAL_ENV"):
        executable = (
            ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
        )
        interpreter = str(Path(os.environ["VIRTUAL_ENV"]).joinpath(*executable))
    else:
        interpreter = sys.executable
    result = subprocess.run(
        [interpreter, "-c", "import mobilerun_functions"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        click.echo(
            f"mobilerun-functions is not installed in {interpreter}. "
            'Install it with: pip install "mobilerun[functions]"',
            err=True,
        )
        raise click.exceptions.Exit(1)
    return interpreter


@click.group(cls=Group, epilog=examples("mobilerun function init function.py"))
@verbose_option
def function_cli(verbose):
    """Manage cloud functions and test functions locally."""


@function_cli.command("init", epilog=examples("mobilerun function init function.py"))
@click.argument("path", type=click.Path(dir_okay=False))
@python_option
@json_option
@verbose_option
def init_function(path, python_path, json_output, verbose):
    """Create a starter function."""
    interpreter = _interpreter(python_path)
    result = subprocess.run(
        [interpreter, "-m", "mobilerun_functions", "new", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if json_output:
        emit({"path": path, "created": result.returncode == 0}, json_output=True)
    elif result.stdout:
        emit(result.stdout.rstrip("\n"))
    if result.stderr:
        click.echo(result.stderr, err=True, nl=False)
    raise click.exceptions.Exit(result.returncode)


def _emit_summary(document):
    # Remote validations do not report it.
    if "entrypoint" in document:
        emit(f"Entry point: {document['entrypoint']}")
    emit(f"Output kind: {document['outputKind']}")
    emit(f"Dataset key: {document['datasetKey']}")
    emit("Input schema:\n" + json.dumps(document["inputSchema"], indent=2))
    emit("Output schema:\n" + json.dumps(document["outputSchema"], indent=2))


def _check_local(path, python_path, json_output):
    interpreter = _interpreter(python_path)
    result = subprocess.run(
        [interpreter, "-m", "mobilerun_functions", "inspect", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stderr:
        click.echo(result.stderr, err=True, nl=False)
    if result.returncode:
        if json_output:
            try:
                json.loads(result.stdout)
            except ValueError:
                pass
            else:
                emit(result.stdout.rstrip("\n"))
        raise click.exceptions.Exit(result.returncode)
    if json_output:
        emit(result.stdout.rstrip("\n"))
    else:
        _emit_summary(json.loads(result.stdout))


def _check_remote(path, wait, api_url, json_output):
    source = _read_source(path)
    with cloud_client(api_url) as client:
        item = client.compute.functions.validations.create(source=source).data
        if wait:
            item = _wait_for_validation(client, item)
    document = item.model_dump(mode="json", by_alias=True)
    pending = item.status == "pending"
    if json_output:
        emit(document, json_output=True)
    elif pending:
        emit(item.id)
    elif item.valid:
        _emit_summary(document)
    if pending:
        click.echo(
            f"Validation still pending; run `mobilerun function check {path} --remote` again.",
            err=True,
        )
        return
    if item.status == "failed":
        print_issues(document if item.errors else {"message": "Validation failed"})
        raise click.exceptions.Exit(1)
    print_issues(document)
    if not item.valid:
        raise click.exceptions.Exit(INVALID_EXIT_CODE)


@function_cli.command(
    "check",
    epilog=examples(
        "mobilerun function check function.py -j",
        "mobilerun function check function.py --remote",
    ),
)
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@python_option
@click.option("--remote", is_flag=True, help="Validate in Mobilerun Cloud.")
@click.option(
    "--wait/--no-wait",
    default=True,
    help=f"With --remote, wait up to {WAIT_SECONDS} seconds.",
)
@api_url_option
@json_option
@verbose_option
def check_function(path, python_path, remote, wait, api_url, json_output, verbose):
    """Inspect a function's entry point and schemas."""
    if remote and python_path:
        raise click.UsageError("--remote and --python cannot be combined.")
    if remote:
        _check_remote(path, wait, api_url, json_output)
    else:
        _check_local(str(path), python_path, json_output)


def _device_flags(
    device, target, platform, api_url
) -> tuple[list[str], dict[str, str] | None]:
    if not (device or target == "cloud"):
        return [], None

    from mobilerun_core.connection.cloud import BEARER_ENV_VAR

    from mobilerun.agent.utils.android_device import is_portal_http_url
    from mobilerun.cli.selection import (
        portal_token,
        require_cloud_api_key,
        select_device,
    )
    from mobilerun.config_manager.config_manager import PORTAL_TOKEN_ENV_VARS

    selection = asyncio.run(
        select_device(device=device, target=target, platform=platform)
    )
    if selection.backend == "cloud":
        return ["--device-target", "cloud", "--device-id", selection.device], {
            **os.environ,
            BEARER_ENV_VAR: require_cloud_api_key(),
            "MOBILERUN_API_BASE_URL": api_url,
        }

    token = portal_token(selection)
    env = None
    if token:
        env = {**os.environ, PORTAL_TOKEN_ENV_VARS[selection.platform]: token}
    device_flag = "--device-id"
    if is_portal_http_url(selection.device):
        device_flag = "--device-url"
    return [
        "--device-target",
        "local",
        "--device-platform",
        selection.platform,
        device_flag,
        selection.device,
    ], env


@function_cli.command(
    "run", epilog=examples("mobilerun function run function.py --local --input '{}'")
)
@click.argument("path", metavar="NAME_OR_ID_OR_FILE")
@click.option("--local", is_flag=True, help="Run a local Python file.")
@click.option("--wait/--no-wait", default=False, help="Wait for the cloud result.")
@click.option("--version", type=click.IntRange(min=1), help="Cloud function version.")
@click.option("--input", "input_json", help="Function input as JSON.")
@click.option(
    "--input-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Read function input from a JSON file.",
)
@click.option(
    "--state-file", type=click.Path(dir_okay=False), help="Read and save function state."
)
@python_option
@device_option
@target_option
@platform_option
@api_url_option
@json_option
@verbose_option
def run_function(
    path,
    local,
    wait,
    version,
    input_json,
    input_file,
    state_file,
    python_path,
    device,
    target,
    platform,
    api_url,
    json_output,
    verbose,
):
    """Run a function and print its result."""
    if (input_json is None) == (input_file is None):
        raise click.UsageError("Provide exactly one of --input or --input-file.")
    try:
        data = json.loads(
            input_json
            if input_json is not None
            else input_file.read_text(encoding="utf-8")
        )
    except (ValueError, OSError) as exc:
        option = "--input" if input_json is not None else "--input-file"
        raise click.BadParameter(str(exc), param_hint=option) from None
    if not local:
        if python_path or state_file or target != "local" or platform:
            raise click.UsageError(
                "--python, --state-file, --target and --platform require --local."
            )
        if device:
            raise click.UsageError(
                "Cloud function runs do not accept --device in SDK 6.3.1."
            )
        if not isinstance(data, dict):
            raise click.BadParameter("Expected a JSON object.", param_hint="--input")
        with cloud_client(api_url) as client:
            item = _resolve_function(client, path)
            params = {"input": data}
            if version is not None:
                params["version"] = version
            run = client.compute.functions.runs.create(item.id, **params).data
            log_download = None
            if wait:
                from mobilerun_sdk import Run, WaitFailureError, WaitTimeoutError

                handle = Run(client, run.id)
                try:
                    run = handle.wait(timeout=WAIT_SECONDS, poll_interval=1).data
                except WaitFailureError as exc:
                    run = exc.response.data
                except WaitTimeoutError:
                    _emit_run(run, json_output)
                    raise click.ClickException(
                        f"Run {run.id} is still running; check `mobilerun function logs {run.id}` later."
                    ) from None
                try:
                    run, log_download = _run_logs(client, run.id)
                except Exception:
                    _emit_run(run, json_output)
                    raise
        _emit_run(run, json_output, log_download=log_download)
        if run.status == "failed":
            raise click.exceptions.Exit(1)
        return
    if version is not None or wait:
        raise click.UsageError("--version and --wait apply to cloud runs only.")
    interpreter = _interpreter(python_path)
    device_args, env = _device_flags(device, target, platform, api_url)
    with tempfile.TemporaryDirectory(prefix="mobilerun-function-") as directory:
        input_path = Path(directory) / "input.json"
        result_path = Path(directory) / "result.json"
        input_path.write_text(json.dumps(data), encoding="utf-8")
        command = [
            interpreter,
            "-m",
            "mobilerun_functions",
            "run",
            path,
            "--input-file",
            str(input_path),
            "--result-file",
            str(result_path),
            *device_args,
        ]
        if state_file is not None:
            command.extend(["--state-file", state_file])
        # Inherit stderr so function logs appear immediately while it runs.
        result = subprocess.run(
            command, stdout=subprocess.PIPE, text=True, check=False, env=env
        )
        if result_path.exists():
            raw = result_path.read_text(encoding="utf-8")
            emit(
                raw.rstrip("\n")
                if json_output
                else json.dumps(json.loads(raw), indent=2)
            )
        elif not result.returncode:
            raise click.ClickException("The function did not write a result file.")
        raise click.exceptions.Exit(result.returncode)


def _functions(client, **filters):
    items = []
    page = 1
    while True:
        result = client.compute.functions.list(page=page, **filters).data
        items.extend(result.items)
        if not result.pagination.has_next:
            return items
        page += 1


def _resolve_function(client, name_or_id):
    from mobilerun_core.detect import is_uuid
    from mobilerun_sdk import NotFoundError

    if is_uuid(name_or_id):
        try:
            return client.compute.functions.retrieve(name_or_id).data
        except NotFoundError:
            pass
    matches = [
        item
        for item in _functions(client, search=name_or_id)
        if item.name == name_or_id
    ]
    if not matches:
        raise click.ClickException(
            f"No function named {name_or_id}; use `mobilerun function list`."
        )
    return matches[0]


def _print_function(item, json_output):
    if json_output:
        emit(item.model_dump(mode="json", by_alias=True), json_output=True)
    else:
        latest = item.latest_version
        emit_rows(
            ("NAME", "VERSION", "ID", "STATUS"),
            [
                (
                    item.name,
                    latest.version if latest else "-",
                    item.id,
                    latest.status if latest else "-",
                )
            ],
        )


def _emit_run(run, json_output, *, log_download=None):
    if json_output:
        document = run.model_dump(mode="json", by_alias=True)
        if log_download is not None:
            document["logDownload"] = log_download.model_dump(
                mode="json", by_alias=True
            )
        emit(document, json_output=True)
    else:
        emit_rows(
            ("ID", "STATUS", "RESULT"), [(run.id, run.status, json.dumps(run.result))]
        )
        if run.stderr:
            click.echo(run.stderr, err=True, nl=False)
        if run.error_code:
            click.echo(run.error_code, err=True)
        if log_download is not None:
            emit(f"Logs: {log_download.url}")


def _poll_until(client, item, fetch, is_final, delays=(1, 2)):
    import httpx
    from mobilerun_sdk import APIError

    # Retried requests could outlive the wait budget.
    poller = client.with_options(max_retries=0)
    deadline = time.monotonic() + WAIT_SECONDS
    delay = delays[0]
    while not is_final(item):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(delay, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            item = fetch(poller, item.id, httpx.Timeout(min(remaining, 10)))
        except APIError:
            # The upload succeeded: retain its identity even if polling fails.
            break
        delay = delays[1]
    return item


def _wait_for_validation(client, item):
    return _poll_until(
        client,
        item,
        lambda poller, item_id, timeout: poller.compute.functions.validations.retrieve(
            item_id, wait_s=VALIDATION_POLL_SECONDS, timeout=timeout
        ).data,
        lambda item: item.status != "pending",
        # The server long-polls (wait_s); pause only between polls so an
        # immediate pending answer cannot become a tight loop.
        delays=(0, 1),
    )


def _read_source(file):
    try:
        return file.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise click.ClickException(f"Cannot read {file}: {exc}") from None


@function_cli.command("push", epilog=examples("mobilerun function push hello.py"))
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--name", help="Function name (defaults to the file basename).")
@click.option("--description", help="Description for a new function.")
@click.option(
    "--manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Manifest JSON with hosts, credentials, and scopes.",
)
@click.option(
    "--wait/--no-wait", default=True, help="Wait up to 60 seconds for inspection."
)
@api_url_option
@json_option
@verbose_option
def push_function(
    file, name, description, manifest, wait, api_url, json_output, verbose
):
    """Create a function or append an immutable version to an existing name."""
    params = {"source": _read_source(file)}
    name = name if name is not None else file.name
    if manifest:
        try:
            params["manifest"] = json.loads(manifest.read_bytes())
        except (ValueError, OSError, UnicodeError) as exc:
            raise click.BadParameter(str(exc), param_hint="--manifest") from None
        if not isinstance(params["manifest"], dict):
            raise click.BadParameter("Expected a JSON object.", param_hint="--manifest")
    with cloud_client(api_url) as client:
        matches = [
            item for item in _functions(client, search=name) if item.name == name
        ]
        if matches:
            item = matches[0]
            if description is not None:
                raise click.UsageError(
                    "--description applies only when creating a function; omit it for a new version."
                )
            client.compute.functions.versions.create(item.id, **params)
            item = client.compute.functions.retrieve(item.id).data
        else:
            if description is not None:
                params["description"] = description
            item = client.compute.functions.create(name=name, **params).data
        if wait:
            item = _poll_until(
                client,
                item,
                lambda poller, item_id, timeout: poller.compute.functions.retrieve(
                    item_id, timeout=timeout
                ).data,
                lambda value: value.latest_version is not None
                and value.latest_version.status != "pending_inspection",
            )
    _print_function(item, json_output)
    latest = item.latest_version
    if latest and latest.status == "rejected":
        print_issues(
            latest.inspect_error.model_dump(mode="json")
            if latest.inspect_error
            else {"message": "Inspection rejected"}
        )
        raise click.exceptions.Exit(1)
    if wait and (latest is None or latest.status == "pending_inspection"):
        click.echo(
            f"Inspection still pending; use `mobilerun function show {item.id}`.",
            err=True,
        )


@function_cli.command("list", epilog=examples("mobilerun function list -j"))
@click.option("--name", help="Filter by exact function name.")
@click.option("--search", help="Search function names.")
@api_url_option
@json_option
@verbose_option
def list_functions(name, search, api_url, json_output, verbose):
    """List functions and their latest versions."""
    with cloud_client(api_url) as client:
        items = _functions(
            client, **({"search": name or search} if name or search else {})
        )
    if name is not None:
        items = [item for item in items if item.name == name]
    if json_output:
        emit(
            [item.model_dump(mode="json", by_alias=True) for item in items],
            json_output=True,
        )
    else:
        emit_rows(
            ("NAME", "VERSION", "ID", "STATUS"),
            (
                (
                    item.name,
                    item.latest_version.version if item.latest_version else "-",
                    item.id,
                    item.latest_version.status if item.latest_version else "-",
                )
                for item in items
            ),
        )


@function_cli.command("show", epilog=examples("mobilerun function show hello.py -j"))
@click.argument("name_or_id")
@click.option(
    "--version", type=click.IntRange(min=1), help="Select a version instead of latest."
)
@api_url_option
@json_option
@verbose_option
def show_function(name_or_id, version, api_url, json_output, verbose):
    """Show function metadata or a selected version."""
    with cloud_client(api_url) as client:
        item = _resolve_function(client, name_or_id)
        if version is not None:
            item = client.compute.functions.versions.retrieve(
                version, function_id=item.id
            ).data
    document = item.model_dump(mode="json", by_alias=True)
    if json_output:
        emit(document, json_output=True)
    else:
        emit_rows(
            ("FIELD", "VALUE"),
            (
                (key, json.dumps(value) if isinstance(value, (dict, list)) else value)
                for key, value in document.items()
            ),
        )


@function_cli.command(
    "versions", epilog=examples("mobilerun function versions hello.py")
)
@click.argument("name_or_id")
@api_url_option
@json_option
@verbose_option
def versions_function(name_or_id, api_url, json_output, verbose):
    """List every version of a function."""
    items = []
    with cloud_client(api_url) as client:
        item = _resolve_function(client, name_or_id)
        page = 1
        while True:
            result = client.compute.functions.versions.list(item.id, page=page).data
            items.extend(value for value in result.items if value is not None)
            if not result.pagination.has_next:
                break
            page += 1
    if json_output:
        emit(
            [value.model_dump(mode="json", by_alias=True) for value in items],
            json_output=True,
        )
    else:
        emit_rows(
            ("VERSION", "STATUS", "CREATED"),
            ((value.version, value.status, value.created_at) for value in items),
        )


@function_cli.command(
    "pull", epilog=examples("mobilerun function pull hello.py -o out.py")
)
@click.argument("name_or_id")
@click.option(
    "--version", type=click.IntRange(min=1), help="Select a version instead of latest."
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Local destination file.",
)
@click.option("--force", is_flag=True, help="Overwrite the destination if it exists.")
@api_url_option
@json_option
@verbose_option
def pull_function(name_or_id, version, output, force, api_url, json_output, verbose):
    """Download source, verifying its SHA-256 digest."""
    path = (
        output
        if output is not None
        else Path(name_or_id if name_or_id.endswith(".py") else name_or_id + ".py")
    )
    if path.exists() and not force:
        raise click.ClickException(
            f"{path} already exists; use --force to overwrite it."
        )
    with cloud_client(api_url) as client:
        item = _resolve_function(client, name_or_id)
        source = client.compute.functions.source(
            item.id, **({"version": version} if version else {})
        ).data
    data = source.source.encode("utf-8")
    if hashlib.sha256(data).hexdigest() != source.sha256:
        raise click.ClickException(
            "Function source SHA-256 does not match the API; no file written."
        )
    write_file(path, data, force=force)
    emit(
        {"id": item.id, "path": str(path)} if json_output else str(path),
        json_output=json_output,
    )


def _run_logs(client, run_id):
    from mobilerun_sdk import Run, WaitFailureError, WaitTimeoutError

    handle = Run(client, run_id)
    try:
        info = handle.wait_logs(timeout=WAIT_SECONDS, poll_interval=1)
    except WaitTimeoutError:
        raise click.ClickException(
            f"Logs for {run_id} are not available yet; retry `mobilerun function logs {run_id}` shortly."
        ) from None
    except WaitFailureError:
        raise click.ClickException(
            f"Logs for {run_id} are unavailable or expired; check the run output instead."
        ) from None
    return info.data, handle.logs().data


@function_cli.command("logs", epilog=examples("mobilerun function logs RUN_ID -j"))
@click.argument("run_id")
@api_url_option
@json_option
@verbose_option
def logs_function(run_id, api_url, json_output, verbose):
    """Wait up to 60 seconds for logs and print their download URL."""
    with cloud_client(api_url) as client:
        _, item = _run_logs(client, run_id)
    emit(
        item.model_dump(mode="json", by_alias=True) if json_output else item.url,
        json_output=json_output,
    )


@function_cli.command(
    "delete", epilog=examples("mobilerun function delete hello.py --all --yes")
)
@click.argument("name_or_id")
@click.option("--version", type=click.IntRange(min=1), help="Delete only this version.")
@click.option(
    "--all",
    "all_versions",
    is_flag=True,
    help="Delete the function and all its versions.",
)
@click.option("--yes", is_flag=True, help="Confirm deletion without prompting.")
@no_input_option
@api_url_option
@json_option
@verbose_option
def delete_function(
    name_or_id, version, all_versions, yes, no_input, api_url, json_output, verbose
):
    """Delete a function, or one of its versions."""
    if version is not None and all_versions:
        raise click.UsageError("Choose --version or --all, not both.")
    if not yes and not input_allowed(no_input):
        raise click.UsageError(
            "Deletion requires --yes when input is unavailable or --no-input is set."
        )
    with cloud_client(api_url) as client:
        item = _resolve_function(client, name_or_id)
        if not yes and not click.confirm(
            f"Delete {item.name} ({"version " + str(version) if version else "all versions"})?",
            err=True,
        ):
            raise click.ClickException("Deletion cancelled.")
        if version is not None:
            client.compute.functions.versions.delete(version, function_id=item.id)
        else:
            client.compute.functions.delete(item.id)
    emit(
        (
            {"id": item.id, "deleted": True, "version": version}
            if json_output
            else f"Deleted {item.name}"
        ),
        json_output=json_output,
    )
