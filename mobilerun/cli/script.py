"""Manage cloud scripts and test sandbox scripts locally."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import click

from mobilerun.cli.options import (
    Group,
    api_url_option,
    emit,
    emit_rows,
    examples,
    input_allowed,
    json_option,
    no_input_option,
    verbose_option,
)

python_option = click.option(
    "--python", "python_path", help="Python interpreter for the script."
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
        [interpreter, "-c", "import mobilerun_sandbox"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        click.echo(
            f"mobilerun-sandbox is not installed in {interpreter}. "
            "Install it with: pip install mobilerun-sandbox",
            err=True,
        )
        raise click.exceptions.Exit(1)
    return interpreter


@click.group(cls=Group, epilog=examples("mobilerun script init script.py"))
@verbose_option
def script_cli(verbose):
    """Manage cloud scripts and test scripts locally."""


@script_cli.command("init", epilog=examples("mobilerun script init script.py"))
@click.argument("path", type=click.Path(dir_okay=False))
@python_option
@verbose_option
def init_script(path, python_path, verbose):
    """Create a starter script."""
    interpreter = _interpreter(python_path)
    result = subprocess.run(
        [interpreter, "-m", "mobilerun_sandbox", "new", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        emit(result.stdout.rstrip("\n"))
    if result.stderr:
        click.echo(result.stderr, err=True, nl=False)
    raise click.exceptions.Exit(result.returncode)


@script_cli.command("check", epilog=examples("mobilerun script check script.py -j"))
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@python_option
@json_option
@verbose_option
def check_script(path, python_path, json_output, verbose):
    """Inspect a script's entry point and schemas."""
    interpreter = _interpreter(python_path)
    result = subprocess.run(
        [interpreter, "-m", "mobilerun_sandbox", "inspect", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stderr:
        click.echo(result.stderr, err=True, nl=False)
    if result.returncode:
        raise click.exceptions.Exit(result.returncode)
    if json_output:
        emit(result.stdout.rstrip("\n"))
    else:
        document = json.loads(result.stdout)
        emit(f"Entry point: {document['entrypoint']}")
        emit(f"Output kind: {document['outputKind']}")
        emit(f"Dataset key: {document['datasetKey']}")
        emit("Input schema:\n" + json.dumps(document["inputSchema"], indent=2))
        emit("Output schema:\n" + json.dumps(document["outputSchema"], indent=2))


@script_cli.command(
    "run", epilog=examples("mobilerun script run script.py --local --input '{}'")
)
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--local", is_flag=True, required=True, help="Run in your local Python environment."
)
@click.option("--input", "input_json", help="Script input as JSON.")
@click.option(
    "--input-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Read script input from a JSON file.",
)
@click.option(
    "--state-file", type=click.Path(dir_okay=False), help="Read and save script state."
)
@python_option
@json_option
@verbose_option
def run_script(
    path, local, input_json, input_file, state_file, python_path, json_output, verbose
):
    """Run a script and print its result."""
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
    interpreter = _interpreter(python_path)
    with tempfile.TemporaryDirectory(prefix="mobilerun-script-") as directory:
        input_path = Path(directory) / "input.json"
        result_path = Path(directory) / "result.json"
        input_path.write_text(json.dumps(data), encoding="utf-8")
        command = [
            interpreter,
            "-m",
            "mobilerun_sandbox",
            "run",
            path,
            "--input-file",
            str(input_path),
            "--result-file",
            str(result_path),
        ]
        if state_file is not None:
            command.extend(["--state-file", state_file])
        # Inherit stderr so script logs appear immediately while it runs.
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=False)
        if result_path.exists():
            raw = result_path.read_text(encoding="utf-8")
            emit(
                raw.rstrip("\n")
                if json_output
                else json.dumps(json.loads(raw), indent=2)
            )
        elif not result.returncode:
            raise click.ClickException("The script did not write a result file.")
        raise click.exceptions.Exit(result.returncode)


@contextmanager
def _script_client(api_url):
    # Keep the SDK and core imports out of help, completion, and version startup.
    from mobilerun_core.connection.cloud import make_sdk_client

    from mobilerun.cli.selection import require_cloud_api_key

    client = make_sdk_client(require_cloud_api_key(), api_url)
    try:
        yield client
    finally:
        client.close()


def _script_versions(client, **filters):
    items = []
    page = 1
    while True:
        result = client.scripts.list(page=page, **filters).data
        items.extend(result.items)
        if not result.pagination.has_next:
            return items
        page += 1


def _named_versions(client, name):
    versions = _script_versions(client, name=name)
    if not versions:
        raise click.ClickException(f"No script named {name}")
    return sorted(versions, key=lambda item: item.version, reverse=True)


def _lookup_script_versions(client, name_or_id):
    from mobilerun_core.detect import is_uuid
    from mobilerun_sdk import NotFoundError

    if is_uuid(name_or_id):
        try:
            return [client.scripts.retrieve(name_or_id).data], True
        except NotFoundError:
            # Canonical UUIDs are also valid script names.
            pass
    return _named_versions(client, name_or_id), False


def _resolve_script(client, name_or_id, version):
    versions, _ = _lookup_script_versions(client, name_or_id)
    return _select_script_version(versions, name_or_id, version)


def _select_script_version(versions, name_or_id, version):
    if version is None:
        return versions[0]
    for item in versions:
        if item.version == version:
            return item
    available = ", ".join(str(item.version) for item in versions)
    raise click.ClickException(
        f"No version {version} of {name_or_id}. Available versions: {available}"
    )


def _script_document(item):
    return item.model_dump(mode="json", by_alias=True)


def _print_script(item, json_output, *, metadata=False):
    document = _script_document(item)
    if json_output:
        emit(document, json_output=True)
    elif metadata:
        emit_rows(
            ("FIELD", "VALUE"),
            (
                (key, json.dumps(value) if isinstance(value, (dict, list)) else value)
                for key, value in document.items()
            ),
        )
    else:
        emit_rows(
            ("NAME", "VERSION", "ID", "STATUS"),
            [(item.name, item.version, item.id, item.status)],
        )


def _wait_for_inspection(client, item):
    import httpx
    from mobilerun_sdk import APIError

    # Retried requests could outlive the inspection wait budget.
    poller = client.with_options(max_retries=0)
    deadline = time.monotonic() + 60
    delay = 1
    while item.status not in {"ready", "rejected"}:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(delay, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            item = poller.scripts.retrieve(
                item.id, timeout=httpx.Timeout(min(remaining, 10))
            ).data
        except APIError:
            # The upload succeeded: retain its identity even if polling fails.
            break
        delay = 2
    return item


@script_cli.command("push", epilog=examples("mobilerun script push hello.py"))
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--name", help="Script name (defaults to the file's basename).")
@click.option("--description", help="Script description.")
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
def push_script(file, name, description, manifest, wait, api_url, json_output, verbose):
    """Upload a new immutable version of a Python script."""
    try:
        source = file.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise click.ClickException(f"Cannot read {file}: {exc}") from None
    params = {"name": name if name is not None else file.name, "source": source}
    if description is not None:
        params["description"] = description
    if manifest is not None:
        try:
            params["manifest"] = json.loads(manifest.read_bytes())
        except (ValueError, OSError, UnicodeError) as exc:
            raise click.BadParameter(str(exc), param_hint="--manifest") from None
        if not isinstance(params["manifest"], dict):
            raise click.BadParameter("Expected a JSON object.", param_hint="--manifest")
    with _script_client(api_url) as client:
        item = client.scripts.create(**params).data
        if wait:
            item = _wait_for_inspection(client, item)
    _print_script(item, json_output)
    if item.status == "rejected":
        error = _script_document(item).get("inspectError")
        click.echo("Inspection rejected: " + json.dumps(error), err=True)
        raise click.exceptions.Exit(1)
    if wait and item.status != "ready":
        click.echo(
            f"Inspection still pending; check with `mobilerun script show {item.name}`.",
            err=True,
        )


@script_cli.command(
    "list",
    epilog=examples(
        "mobilerun script list", "mobilerun script list --name hello.py -j"
    ),
)
@click.option("--name", help="List all versions of this exact name.")
@click.option("--search", help="Search script names.")
@api_url_option
@json_option
@verbose_option
def list_scripts(name, search, api_url, json_output, verbose):
    """List the latest version per name, or all versions of one name."""
    filters = {"latest": "true"} if name is None else {"name": name}
    if search is not None:
        filters["search"] = search
    with _script_client(api_url) as client:
        items = _script_versions(client, **filters)
    if name is not None:
        items.sort(key=lambda item: item.version, reverse=True)
    if json_output:
        emit([_script_document(item) for item in items], json_output=True)
    else:
        emit_rows(
            ("NAME", "VERSION", "VERSIONS", "STATUS", "CREATED"),
            (
                (
                    item.name,
                    item.version,
                    item.version_count,
                    item.status,
                    item.created_at.isoformat(),
                )
                for item in items
            ),
        )


@script_cli.command(
    "show", epilog=examples("mobilerun script show hello.py --version 1 -j")
)
@click.argument("name_or_id")
@click.option(
    "--version", type=click.IntRange(min=1), help="Select a version instead of latest."
)
@api_url_option
@json_option
@verbose_option
def show_script(name_or_id, version, api_url, json_output, verbose):
    """Show script metadata, manifest, schemas, and inspection errors."""
    with _script_client(api_url) as client:
        item = _resolve_script(client, name_or_id, version)
    _print_script(item, json_output, metadata=True)


@script_cli.command(
    "pull", epilog=examples("mobilerun script pull hello.py --version 1 -o out.py")
)
@click.argument("name_or_id")
@click.option(
    "--version", type=click.IntRange(min=1), help="Select a version instead of latest."
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination file.",
)
@click.option("--force", is_flag=True, help="Overwrite the destination if it exists.")
@api_url_option
@json_option
@verbose_option
def pull_script(name_or_id, version, output, force, api_url, json_output, verbose):
    """Download verified source to a local file."""
    # Never use a server-supplied name as a local path.
    path = (
        output
        if output is not None
        else Path(name_or_id if name_or_id.endswith(".py") else name_or_id + ".py")
    )
    if path.exists() and not force:
        raise click.ClickException(
            f"{path} already exists; use --force to overwrite it."
        )
    with _script_client(api_url) as client:
        item = _resolve_script(client, name_or_id, version)
        source = client.scripts.source(item.id).data
    data = source.source.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    if digest != source.sha256:
        raise click.ClickException(
            "Script source SHA-256 does not match the API; no file written."
        )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".mobilerun-pull-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            os.replace(temporary, path)
        else:
            # Linking publishes the complete file only if the destination is free.
            os.link(temporary, path)
    except FileExistsError:
        raise click.ClickException(
            f"{path} already exists; use --force to overwrite it."
        ) from None
    except OSError as exc:
        raise click.ClickException(f"Cannot write {path}: {exc}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    document = {
        "id": item.id,
        "name": item.name,
        "version": item.version,
        "path": str(path),
        "sha256": digest,
        "sizeBytes": len(data),
    }
    emit(document if json_output else str(path), json_output=json_output)


@script_cli.command(
    "delete",
    epilog=examples(
        "mobilerun script delete hello.py --version 1 --yes",
        "mobilerun script delete hello.py --all --yes -j",
    ),
)
@click.argument("name_or_id")
@click.option("--version", type=click.IntRange(min=1), help="Delete this version.")
@click.option(
    "--all", "all_versions", is_flag=True, help="Delete every active version of a name."
)
@click.option("--yes", is_flag=True, help="Confirm deletion without prompting.")
@no_input_option
@api_url_option
@json_option
@verbose_option
def delete_script(
    name_or_id, version, all_versions, yes, no_input, api_url, json_output, verbose
):
    """Delete one script version, or all versions of a name."""
    from mobilerun_core.detect import is_uuid

    if version is not None and all_versions:
        raise click.UsageError("Choose --version or --all, not both.")
    if not is_uuid(name_or_id) and version is None and not all_versions:
        raise click.UsageError("A script name requires --version N or --all.")
    if not yes and not input_allowed(no_input):
        raise click.UsageError(
            "Deletion requires --yes when input is unavailable or --no-input is set."
        )
    with _script_client(api_url) as client:
        versions, by_id = _lookup_script_versions(client, name_or_id)
        if by_id and all_versions:
            raise click.UsageError("--all requires a script name, not an id.")
        if not by_id and version is None and not all_versions:
            raise click.UsageError("A script name requires --version N or --all.")
        items = (
            versions
            if all_versions
            else [_select_script_version(versions, name_or_id, version)]
        )
        if not yes and not click.confirm(
            f"Delete {len(items)} version(s) of {name_or_id}?", err=True
        ):
            raise click.ClickException("Deletion cancelled.")
        for item in items:
            client.scripts.delete(item.id)
    documents = [
        {"id": item.id, "name": item.name, "version": item.version} for item in items
    ]
    if json_output:
        emit({"deleted": documents}, json_output=True)
    else:
        emit_rows(
            ("NAME", "VERSION", "ID"),
            ((item.name, item.version, item.id) for item in items),
        )
