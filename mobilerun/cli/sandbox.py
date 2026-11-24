"""Manage cloud sandboxes through the public compute SDK."""

import time
from copy import copy
from pathlib import Path

import click

from mobilerun.cli.options import (
    TIMEOUT_EXIT_CODE,
    Group,
    all_pages,
    api_url_option,
    cloud_client,
    confirm_destructive,
    emit,
    emit_rows,
    examples,
    json_option,
    no_input_option,
    print_models,
    refuse_existing,
    verbose_option,
    write_file,
)

# Server-side long-poll for the first exec response, and the extra wait for it to finish.
EXEC_LONG_POLL_SECONDS = 30
EXEC_WAIT_GRACE_SECONDS = 30
# Long-poll per status request while the command runs.
EXEC_POLL_SECONDS = 5
SANDBOX_HEADERS = ("ID", "NAME", "STATUS", "EXPIRES")


@click.group(cls=Group, epilog=examples("mobilerun sandbox create --timeout 300"))
@verbose_option
def sandbox_cli(verbose):
    """Create sandboxes, execute commands, and transfer files."""


def _sandbox_row(item):
    return (item.id, item.name, item.status, item.expires_at)


def _show(item, json_output):
    if json_output:
        emit(item.model_dump(mode="json", by_alias=True), json_output=True)
    else:
        emit_rows(SANDBOX_HEADERS, [_sandbox_row(item)])


def _env(values):
    result = {}
    for value in values:
        key, separator, data = value.partition("=")
        if not separator or not key:
            raise click.BadParameter("Use NAME=VALUE.", param_hint="--env")
        result[key] = data
    return result


@sandbox_cli.command(
    "create",
    epilog=examples(
        "mobilerun sandbox create --name demo --timeout 300 --env MODE=test"
    ),
)
@click.option("--name", help="Sandbox name.")
@click.option(
    "--timeout",
    "timeout_s",
    type=click.IntRange(min=1),
    help="Maximum sandbox lifetime in seconds.",
)
@click.option(
    "--idle-timeout",
    "idle_timeout_s",
    type=click.IntRange(min=1),
    help="Stop after this many idle seconds.",
)
@click.option(
    "--env", multiple=True, help="Environment variable NAME=VALUE (repeatable)."
)
@api_url_option
@json_option
@verbose_option
def create_sandbox(name, timeout_s, idle_timeout_s, env, api_url, json_output, verbose):
    """Create a sandbox."""
    params = {
        key: value
        for key, value in {
            "name": name,
            "timeout_s": timeout_s,
            "idle_timeout_s": idle_timeout_s,
        }.items()
        if value is not None
    }
    if env:
        params["env"] = _env(env)
    with cloud_client(api_url) as client:
        item = client.compute.sandboxes.create(**params).data
    _show(item, json_output)


@sandbox_cli.command("list", epilog=examples("mobilerun sandbox list -j"))
@api_url_option
@json_option
@verbose_option
def list_sandboxes(api_url, json_output, verbose):
    """List sandboxes, including stopped sandboxes retained by the API."""
    with cloud_client(api_url) as client:
        items = all_pages(lambda page: client.compute.sandboxes.list(page=page).data)
    print_models(items, json_output, SANDBOX_HEADERS, _sandbox_row)


@sandbox_cli.command("show", epilog=examples("mobilerun sandbox show ID"))
@click.argument("sandbox_id", metavar="ID")
@api_url_option
@json_option
@verbose_option
def show_sandbox(sandbox_id, api_url, json_output, verbose):
    """Show a sandbox."""
    with cloud_client(api_url) as client:
        item = client.compute.sandboxes.retrieve(sandbox_id).data
    _show(item, json_output)


@sandbox_cli.command(
    "exec", epilog=examples("mobilerun sandbox exec ID \"python -c 'print(1)'\"")
)
@click.argument("sandbox_id", metavar="ID")
@click.argument("command")
@click.option("--cwd", help="Working directory in the sandbox.")
@click.option(
    "--env", multiple=True, help="Environment variable NAME=VALUE (repeatable)."
)
@click.option(
    "--timeout",
    "timeout_s",
    type=click.IntRange(min=1),
    default=60,
    help="Command deadline in seconds; exits 124 if execution or its wait times out.",
)
@api_url_option
@json_option
@verbose_option
def exec_sandbox(
    sandbox_id, command, cwd, env, timeout_s, api_url, json_output, verbose
):
    """Execute a shell command and return its remote exit code."""
    params = {
        "command": command,
        "timeout_s": timeout_s,
        "wait_s": min(timeout_s, EXEC_LONG_POLL_SECONDS),
    }
    if cwd is not None:
        params["cwd"] = cwd
    if env:
        params["env"] = _env(env)
    with cloud_client(api_url) as client:
        result = client.compute.sandboxes.exec(sandbox_id, **params).data
        deadline = time.monotonic() + timeout_s + EXEC_WAIT_GRACE_SECONDS
        while result.status == "running" and time.monotonic() < deadline:
            result = client.compute.sandboxes.commands.retrieve(
                result.command_id, sandbox_id=sandbox_id, wait_s=EXEC_POLL_SECONDS
            ).data
    if json_output:
        emit(result.model_dump(mode="json", by_alias=True), json_output=True)
    else:
        if result.stdout:
            emit(result.stdout.rstrip("\n"))
        if result.stderr:
            click.echo(result.stderr, err=True, nl=False)
    if result.status == "running":
        click.echo(
            f"Command {result.command_id} is still running and may still finish "
            "in the sandbox; the wait timed out and its exit code is not available.",
            err=True,
        )
        raise click.exceptions.Exit(TIMEOUT_EXIT_CODE)
    if result.timed_out:
        click.echo("Command timed out.", err=True)
        raise click.exceptions.Exit(TIMEOUT_EXIT_CODE)
    if result.truncated:
        click.echo("Command output was truncated by the API.", err=True)
    raise click.exceptions.Exit(result.exit_code if result.exit_code is not None else 1)


@sandbox_cli.group(
    "files",
    cls=Group,
    epilog=examples("mobilerun sandbox files upload ID local.txt remote.txt"),
)
@verbose_option
def sandbox_files(verbose):
    """Upload and download files (the API has no file listing endpoint)."""


@sandbox_files.command(
    "upload",
    epilog=examples("mobilerun sandbox files upload ID local.txt remote.txt"),
)
@click.argument("sandbox_id", metavar="ID")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("path")
@api_url_option
@json_option
@verbose_option
def upload_file(sandbox_id, file, path, api_url, json_output, verbose):
    """Upload a local file inside the sandbox work directory (relative path allowed)."""
    with cloud_client(api_url) as client:
        client.compute.sandboxes.files.upload(sandbox_id, file, path=path)
    emit({"path": path} if json_output else path, json_output=json_output)


@sandbox_files.command(
    "download",
    epilog=examples("mobilerun sandbox files download ID remote.txt -o local.txt"),
)
@click.argument("sandbox_id", metavar="ID")
@click.argument("path")
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Local destination file.",
)
@click.option(
    "--force", is_flag=True, help="Overwrite the local destination if it exists."
)
@api_url_option
@json_option
@verbose_option
def download_file(sandbox_id, path, output, force, api_url, json_output, verbose):
    """Download a file from the sandbox work directory."""
    refuse_existing(output, force)
    with cloud_client(api_url) as client:
        response = client.compute.sandboxes.files.retrieve(sandbox_id, path=path)
        data = response.read()
    write_file(output, data, force=force)
    emit({"path": str(output)} if json_output else str(output), json_output=json_output)


@sandbox_cli.command("delete", epilog=examples("mobilerun sandbox delete ID --yes"))
@click.argument("sandbox_id", metavar="ID")
@click.option("--yes", is_flag=True, help="Confirm stopping the sandbox.")
@no_input_option
@api_url_option
@json_option
@verbose_option
def delete_sandbox(sandbox_id, yes, no_input, api_url, json_output, verbose):
    """Delete (stop) a sandbox."""
    confirm_destructive(f"Stop sandbox {sandbox_id}?", yes=yes, no_input=no_input)
    with cloud_client(api_url) as client:
        client.compute.sandboxes.delete(sandbox_id)
    emit(
        {"id": sandbox_id, "stopped": True} if json_output else f"Stopped {sandbox_id}",
        json_output=json_output,
    )
    if not json_output:
        click.echo(
            "Deleted sandboxes may still appear as stopped in `mobilerun sandbox list`.",
            err=True,
        )


# Keep the old spelling callable without adding a second advertised operation.
stop_sandbox = copy(delete_sandbox)
stop_sandbox.name = "stop"
stop_sandbox.hidden = True
sandbox_cli.add_command(stop_sandbox)
