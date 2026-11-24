"""Validate, push and pull routine documents in Mobilerun Cloud."""

from __future__ import annotations

import json
from pathlib import Path

import click

from mobilerun.cli.documents import read_document, render_document
from mobilerun.cli.options import (
    INVALID_EXIT_CODE,
    Group,
    all_pages,
    api_url_option,
    cloud_client,
    confirm_destructive,
    convert_api_error,
    emit,
    emit_rows,
    examples,
    json_option,
    no_input_option,
    print_issues,
    print_model,
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


def _push_document(client, flow_id, document, json_output):
    """Create or replace a routine; render a rejected document as issues."""
    from mobilerun_sdk import APIStatusError

    flows = client.workflows.flows
    try:
        if flow_id is None:
            return flows.create_from_document(document=document).data
        return flows.replace_document(flow_id, document=document).data
    except APIStatusError as exc:
        # The converted document redacts secrets; `-j` prints it as for other errors.
        _, error, _ = convert_api_error(exc)
        details = error.get("details")
        if exc.status_code != DOCUMENT_REJECTED_STATUS or not isinstance(details, list):
            raise
        if json_output:
            emit(error, json_output=True)
        print_issues({"errors": details})
        raise click.exceptions.Exit(INVALID_EXIT_CODE) from None


def _emit_or_write(document, output, force, json_output, format="json"):
    text = render_document(document, "json" if json_output else format)
    if output is None:
        emit(
            json.loads(text) if json_output else text.rstrip("\n"),
            json_output=json_output,
        )
        return
    write_file(output, text.encode("utf-8"), force=force)
    emit({"path": str(output)} if json_output else str(output), json_output=json_output)


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
    _emit_or_write(schema, output, force, json_output)


@routine_cli.command(
    "check", epilog=examples("mobilerun routine check routine.json -j")
)
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@api_url_option
@json_option
@verbose_option
def check_routine(file, api_url, json_output, verbose):
    """Validate a routine document in Mobilerun Cloud."""
    document = read_document(file)
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
    document = read_document(file)
    with cloud_client(api_url) as client:
        result = _push_document(client, flow_id, document, json_output)
    flow = result.flow
    if json_output:
        emit(result.model_dump(mode="json", by_alias=True), json_output=True)
        return
    emit_rows(("ID", "NAME", "STATUS"), [(flow.id, flow.name, flow.lifecycle_status)])
    if flow_id is None:
        click.echo(
            "The routine is created disabled; enable it with `mobilerun routine enable ID`.",
            err=True,
        )


@routine_cli.command(
    "pull", epilog=examples("mobilerun routine pull ID -o routine.yaml")
)
@click.argument("flow_id", metavar="ID")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the document to this file instead of stdout.",
)
@click.option("--force", is_flag=True, help="Overwrite the output file if it exists.")
@click.option(
    "--format",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    help="Document output format (--json overrides this).",
)
@api_url_option
@json_option
@verbose_option
def pull_routine(flow_id, output, force, format, api_url, json_output, verbose):
    """Download a routine as a document."""
    with cloud_client(api_url) as client:
        document = client.workflows.flows.retrieve_document(flow_id).data
    _emit_or_write(document, output, force, json_output, format)


@routine_cli.command("list", epilog=examples("mobilerun routine list -j"))
@api_url_option
@json_option
@verbose_option
def list_routines(api_url, json_output, verbose):
    """List routines."""
    with cloud_client(api_url) as client:
        items = all_pages(lambda page: client.workflows.flows.list(page=page))
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


@routine_cli.command("show", epilog=examples("mobilerun routine show ID -j"))
@click.argument("flow_id", metavar="ID")
@api_url_option
@json_option
@verbose_option
def show_routine(flow_id, api_url, json_output, verbose):
    """Show routine metadata."""
    with cloud_client(api_url) as client:
        item = client.workflows.flows.retrieve(flow_id).data
    print_model(item, json_output)


@routine_cli.command("run", epilog=examples("mobilerun routine run ID --input '{}'"))
@click.argument("flow_id", metavar="ID")
@click.option(
    "--input", "input_json", default="{}", help="Event payload as a JSON object."
)
@api_url_option
@json_option
@verbose_option
def run_routine(flow_id, input_json, api_url, json_output, verbose):
    """Run an enabled routine immediately, bypassing trigger conditions."""
    try:
        payload = json.loads(input_json)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--input") from None
    if not isinstance(payload, dict):
        raise click.BadParameter("Expected a JSON object.", param_hint="--input")
    with cloud_client(api_url) as client:
        result = client.workflows.flows.run(flow_id, payload=payload).data
    print_model(result, json_output)
    if result.skipped:
        click.echo(
            f"Run skipped: {result.skipped.reason}; check `mobilerun routine show {flow_id}`.",
            err=True,
        )
        raise click.exceptions.Exit(1)
    if result.enqueue_failed:
        raise click.ClickException(
            "The API could not enqueue all executions; check `mobilerun routine runs ID`."
        )


@routine_cli.command("runs", epilog=examples("mobilerun routine runs ID -j"))
@click.argument("flow_id", metavar="ID")
@api_url_option
@json_option
@verbose_option
def routine_runs(flow_id, api_url, json_output, verbose):
    """List a routine's executions."""
    with cloud_client(api_url) as client:
        items = all_pages(
            lambda page: client.workflows.executions.list(flow_id=flow_id, page=page)
        )
    if json_output:
        emit(
            [item.model_dump(mode="json", by_alias=True) for item in items],
            json_output=True,
        )
    else:
        emit_rows(
            ("ID", "STATUS", "STARTED"),
            ((item.id, item.status, item.started_at) for item in items),
        )


@routine_cli.command("enable", epilog=examples("mobilerun routine enable ID"))
@click.argument("flow_id", metavar="ID")
@api_url_option
@json_option
@verbose_option
def enable_routine(flow_id, api_url, json_output, verbose):
    """Enable a routine; the API decides whether activation is allowed."""
    with cloud_client(api_url) as client:
        item = client.workflows.flows.update(flow_id, lifecycle_status="enabled").data
    print_model(item, json_output)


@routine_cli.command("disable", epilog=examples("mobilerun routine disable ID"))
@click.argument("flow_id", metavar="ID")
@api_url_option
@json_option
@verbose_option
def disable_routine(flow_id, api_url, json_output, verbose):
    """Disable a routine."""
    with cloud_client(api_url) as client:
        item = client.workflows.flows.update(flow_id, lifecycle_status="disabled").data
    print_model(item, json_output)


@routine_cli.command("delete", epilog=examples("mobilerun routine delete ID --yes"))
@click.argument("flow_id", metavar="ID")
@click.option("--yes", is_flag=True, help="Confirm deletion without prompting.")
@no_input_option
@api_url_option
@json_option
@verbose_option
def delete_routine(flow_id, yes, no_input, api_url, json_output, verbose):
    """Delete a routine."""
    confirm_destructive(f"Delete routine {flow_id}?", yes=yes, no_input=no_input)
    with cloud_client(api_url) as client:
        client.workflows.flows.delete(flow_id)
    emit(
        {"id": flow_id, "deleted": True} if json_output else f"Deleted {flow_id}",
        json_output=json_output,
    )
