"""Validate, push and pull routine documents in Mobilerun Cloud."""

from __future__ import annotations

import json
from pathlib import Path

import click

from mobilerun.cli.options import (
    INVALID_EXIT_CODE,
    Group,
    api_url_option,
    cloud_client,
    emit,
    emit_rows,
    examples,
    json_option,
    print_issues,
    verbose_option,
    write_file,
)

DOCUMENT_REJECTED_STATUS = 422


@click.group(
    cls=Group,
    epilog=examples("mobilerun routine schema", "mobilerun routine check routine.json"),
)
@verbose_option
def routine_cli(verbose):
    """Validate and push routine documents."""


def _read_document(file):
    try:
        return json.loads(file.read_bytes())
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Cannot read {file} as JSON: {exc}") from None


def _push_document(client, flow_id, document, json_output):
    """Create or replace a routine; render a rejected document, not raw error JSON."""
    from mobilerun_sdk import APIStatusError

    flows = client.workflows.flows
    try:
        if flow_id is None:
            return flows.create_from_document(document=document).data
        return flows.replace_document(flow_id, document=document).data
    except APIStatusError as exc:
        body = exc.body if isinstance(exc.body, dict) else {}
        details = body.get("details")
        if exc.status_code != DOCUMENT_REJECTED_STATUS or not isinstance(details, list):
            raise
        if json_output:
            emit(body, json_output=True)
        print_issues({"errors": details})
        raise click.exceptions.Exit(INVALID_EXIT_CODE) from None


@routine_cli.command(
    "schema", epilog=examples("mobilerun routine schema -o schema.json")
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the schema to this file.",
)
@click.option("--force", is_flag=True, help="Overwrite the output file if it exists.")
@api_url_option
@json_option
@verbose_option
def routine_schema(output, force, api_url, json_output, verbose):
    """Print the JSON schema of routine documents."""
    with cloud_client(api_url) as client:
        schema = client.workflows.flows.document_schema()
    text = json.dumps(schema, indent=2)
    if output is None:
        emit(schema if json_output else text, json_output=json_output)
        return
    write_file(output, (text + "\n").encode("utf-8"), force=force)
    emit({"path": str(output)} if json_output else str(output), json_output=json_output)


@routine_cli.command(
    "check", epilog=examples("mobilerun routine check routine.json -j")
)
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@api_url_option
@json_option
@verbose_option
def check_routine(file, api_url, json_output, verbose):
    """Validate a routine document in Mobilerun Cloud."""
    document = _read_document(file)
    with cloud_client(api_url) as client:
        result = client.workflows.flows.validate_document(document=document).data
    verdict = result.model_dump(mode="json", by_alias=True)
    if json_output:
        emit(verdict, json_output=True)
    elif result.valid:
        emit("valid")
    print_issues(verdict)
    if not result.valid:
        raise click.exceptions.Exit(INVALID_EXIT_CODE)


@routine_cli.command(
    "push",
    epilog=examples(
        "mobilerun routine push routine.json",
        "mobilerun routine push routine.json --id ID",
    ),
)
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--id", "flow_id", help="Replace this routine instead of creating one.")
@api_url_option
@json_option
@verbose_option
def push_routine(file, flow_id, api_url, json_output, verbose):
    """Create a routine from a document, or replace one with --id."""
    document = _read_document(file)
    with cloud_client(api_url) as client:
        result = _push_document(client, flow_id, document, json_output)
    flow = result.flow
    if json_output:
        emit(result.model_dump(mode="json", by_alias=True), json_output=True)
        return
    emit_rows(("ID", "NAME", "STATUS"), [(flow.id, flow.name, flow.lifecycle_status)])
    if flow_id is None:
        click.echo(
            "The routine is created disabled; activate it in Mobilerun Cloud.", err=True
        )


@routine_cli.command(
    "pull", epilog=examples("mobilerun routine pull ID -o routine.json")
)
@click.argument("flow_id", metavar="ID")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the document to this file instead of stdout.",
)
@click.option("--force", is_flag=True, help="Overwrite the output file if it exists.")
@api_url_option
@json_option
@verbose_option
def pull_routine(flow_id, output, force, api_url, json_output, verbose):
    """Download a routine as a document."""
    if output is not None and output.exists() and not force:
        raise click.ClickException(
            f"{output} already exists; use --force to overwrite it."
        )
    with cloud_client(api_url) as client:
        document = client.workflows.flows.retrieve_document(flow_id).data
    text = json.dumps(document, indent=2)
    if output is None:
        emit(document if json_output else text, json_output=json_output)
        return
    write_file(output, (text + "\n").encode("utf-8"), force=force)
    emit({"path": str(output)} if json_output else str(output), json_output=json_output)


def _routines(client):
    items = []
    page = 1
    while True:
        result = client.workflows.flows.list(page=page)
        items.extend(result.items)
        if not result.pagination.has_next:
            return items
        page += 1


@routine_cli.command("list", epilog=examples("mobilerun routine list -j"))
@api_url_option
@json_option
@verbose_option
def list_routines(api_url, json_output, verbose):
    """List routines."""
    with cloud_client(api_url) as client:
        items = _routines(client)
    if json_output:
        emit(
            [item.model_dump(mode="json", by_alias=True) for item in items],
            json_output=True,
        )
        return
    emit_rows(
        ("ID", "NAME", "STATUS", "UPDATED"),
        (
            (
                item.id,
                item.name,
                item.lifecycle_status,
                item.updated_at.isoformat() if item.updated_at else "-",
            )
            for item in items
        ),
    )
