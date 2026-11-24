"""Manage cloud sandboxes through the public compute SDK."""

import time
from pathlib import Path

import click

from mobilerun.cli.options import (
    Group,
    api_url_option,
    cloud_client,
    emit,
    emit_rows,
    examples,
    json_option,
    verbose_option,
    write_file,
)


@click.group(cls=Group, epilog=examples("mobilerun sandbox create --timeout 300"))
@verbose_option
def sandbox_cli(verbose):
    """Create sandboxes, execute commands, and transfer files."""


def _show(item, json_output):
    if json_output:
        emit(item.model_dump(mode="json", by_alias=True), json_output=True)
    else:
        emit_rows(
            ("ID", "NAME", "STATUS", "EXPIRES"),
            [(item.id, item.name, item.status, item.expires_at)],
        )


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
    items = []
    with cloud_client(api_url) as client:
        page = 1
        while True:
            result = client.compute.sandboxes.list(page=page).data
            items.extend(result.items)
            if not result.pagination.has_next:
                break
            page += 1
    if json_output:
        emit(
            [item.model_dump(mode="json", by_alias=True) for item in items],
            json_output=True,
        )
    else:
        emit_rows(
            ("ID", "NAME", "STATUS", "EXPIRES"),
            ((item.id, item.name, item.status, item.expires_at) for item in items),
        )


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
    help="Command deadline in seconds.",
)
@api_url_option
@json_option
@verbose_option
def exec_sandbox(
    sandbox_id, command, cwd, env, timeout_s, api_url, json_output, verbose
):
    """Execute a shell command and return its remote exit code."""
    params = {"command": command, "timeout_s": timeout_s, "wait_s": min(timeout_s, 30)}
    if cwd is not None:
        params["cwd"] = cwd
    if env:
        params["env"] = _env(env)
    with cloud_client(api_url) as client:
        result = client.compute.sandboxes.exec(sandbox_id, **params).data
        deadline = time.monotonic() + timeout_s + 30
        while result.status == "running" and time.monotonic() < deadline:
            result = client.compute.sandboxes.commands.retrieve(
                result.command_id, sandbox_id=sandbox_id, wait_s=5
            ).data
    if json_output:
        emit(result.model_dump(mode="json", by_alias=True), json_output=True)
    else:
        emit(result.stdout.rstrip("\n")) if result.stdout else None
        if result.stderr:
            click.echo(result.stderr, err=True, nl=False)
    if result.status == "running":
        raise click.ClickException(
            f"Command {result.command_id} is still running; its exit code is not available."
        )
    if result.timed_out:
        click.echo("Command timed out.", err=True)
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
    with cloud_client(api_url) as client, file.open("rb") as stream:
        client.compute.sandboxes.files.upload(sandbox_id, stream, path=path)
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
    if output.exists() and not force:
        raise click.ClickException(
            f"{output} already exists; use --force to overwrite it."
        )
    with cloud_client(api_url) as client:
        response = client.compute.sandboxes.files.retrieve(sandbox_id, path=path)
        data = response.read()
    write_file(output, data, force=force)
    emit({"path": str(output)} if json_output else str(output), json_output=json_output)


def _stop(sandbox_id, yes, api_url, json_output):
    if not yes:
        raise click.UsageError("Stopping a sandbox requires --yes.")
    with cloud_client(api_url) as client:
        client.compute.sandboxes.delete(sandbox_id)
    emit(
        {"id": sandbox_id, "stopped": True} if json_output else f"Stopped {sandbox_id}",
        json_output=json_output,
    )
    click.echo(
        "Deleted sandboxes may still appear as stopped in `mobilerun sandbox list`.",
        err=True,
    )


@sandbox_cli.command("stop", epilog=examples("mobilerun sandbox stop ID --yes"))
@click.argument("sandbox_id", metavar="ID")
@click.option("--yes", is_flag=True, help="Confirm stopping the sandbox.")
@api_url_option
@json_option
@verbose_option
def stop_sandbox(sandbox_id, yes, api_url, json_output, verbose):
    """Stop a sandbox using the API's delete operation."""
    _stop(sandbox_id, yes, api_url, json_output)


@sandbox_cli.command("delete", epilog=examples("mobilerun sandbox delete ID --yes"))
@click.argument("sandbox_id", metavar="ID")
@click.option("--yes", is_flag=True, help="Confirm stopping the sandbox.")
@api_url_option
@json_option
@verbose_option
def delete_sandbox(sandbox_id, yes, api_url, json_output, verbose):
    """Delete (stop) a sandbox."""
    _stop(sandbox_id, yes, api_url, json_output)
