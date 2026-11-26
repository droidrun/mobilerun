"""Manage webhook endpoints and inspect destinations with query values masked."""

import asyncio
import json
from http import HTTPStatus

import click
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mobilerun.cli.base import render
from mobilerun.cli.base.menu import MenuGroup, is_interactive
from mobilerun.cli.base.prompts import SelectChoice, confirm_prompt, select_prompt
from mobilerun.core.webhooks import (
    DEFAULT_WEBHOOK_EVENT,
    CloudWebhooks,
    WebhookEndpoint,
    validate_webhook_url,
)
from mobilerun.errors.cloud import CloudConflict, CloudValidationError
from mobilerun.utils.redaction import mask_url


def _endpoint_url(endpoint: WebhookEndpoint) -> str | None:
    return (
        mask_url(endpoint.url.get_secret_value()) if endpoint.url is not None else None
    )


def _endpoint_id(value: str | None) -> str:
    if value:
        return value
    if not is_interactive():
        raise click.UsageError("Missing required ID.")
    endpoints = asyncio.run(CloudWebhooks().list())
    if not endpoints:
        click.echo("No webhooks. Create one: mobilerun webhook register <url>")
        raise click.exceptions.Exit(0)
    choices = [
        SelectChoice(
            e.id,
            f"{e.id[:8]}  {_endpoint_url(e) or e.kind}  {', '.join(e.event_types) or 'all'}  {e.state}  {e.health}",
        )
        for e in endpoints
    ]
    selected = select_prompt("Select a webhook", choices)
    if selected is None:
        raise click.Abort()
    return selected


def _destination(url: str | None) -> str:
    while True:
        if url is None and not is_interactive():
            raise click.UsageError("Missing required webhook URL.")
        value = (
            url
            if url is not None
            else click.prompt("Webhook URL (https://…)", err=True)
        )
        try:
            return validate_webhook_url(value)
        except CloudValidationError as exc:
            reason = str(exc)
        if not is_interactive():
            raise click.UsageError(reason)
        click.echo(reason, err=True)
        url = None


def _metadata(endpoint: WebhookEndpoint) -> dict[str, object]:
    return {
        "id": endpoint.id,
        "kind": endpoint.kind,
        "state": endpoint.state,
        "health": endpoint.health,
        "url": _endpoint_url(endpoint),
        "event_types": endpoint.event_types,
    }


@click.group(cls=MenuGroup)
def webhook() -> None:
    """Manage cloud webhook endpoints."""


@webhook.command()
@click.argument("url", required=False)
@click.option(
    "--event-type",
    "event_types",
    multiple=True,
    default=(DEFAULT_WEBHOOK_EVENT,),
    help=f"Default: flow results ({DEFAULT_WEBHOOK_EVENT}). Repeat to specify other events.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output JSON, including the one-time secret on creation.",
)
def register(url: str | None, event_types: tuple[str, ...], as_json: bool) -> None:
    """Register URL unless a matching endpoint already exists."""
    destination = _destination(url)
    hooks = CloudWebhooks()
    endpoint = asyncio.run(hooks.find_by_url(destination))
    created = endpoint is None
    if endpoint is None:
        try:
            endpoint = asyncio.run(hooks.register(destination, event_types=event_types))
        except CloudConflict as exc:
            if exc.code != "webhook_duplicate_url":
                raise
            endpoint = asyncio.run(hooks.find_by_url(destination))
            if endpoint is None:
                raise
            created = False
    data = _metadata(endpoint)
    data["created"] = created
    status = "Registered webhook" if created else "Webhook already registered"
    click.echo(f"{status}: {endpoint.id}", err=as_json)
    receives_results = (
        not endpoint.event_types or DEFAULT_WEBHOOK_EVENT in endpoint.event_types
    )
    if not created:
        missing_events = (
            set(event_types).difference(endpoint.event_types)
            if endpoint.event_types
            else set()
        )
        reason = None
        if endpoint.health == "blocked":
            reason = "the endpoint is blocked"
        elif endpoint.state != "ACTIVE":
            reason = f"the endpoint is {endpoint.state.lower()}"
        elif missing_events:
            reason = f"its filter excludes {', '.join(sorted(missing_events))}"
        if reason:
            message = f"Cannot reuse webhook: {reason}."
            click.echo(message, err=as_json)
            if as_json:
                click.echo(json.dumps({**data, "error": message}, ensure_ascii=False))
            raise click.exceptions.Exit(1)
    if created and endpoint.secret is not None:
        notice = "Signing secret (shown once, store it now)"
        if as_json:
            click.echo(notice, err=True)
            data["secret"] = endpoint.secret.get_secret_value()
        else:
            click.echo(f"{notice}: {endpoint.secret.get_secret_value()}")
    elif created:
        click.echo(
            f'Signing secret unavailable (already issued); use client.webhooks.rotate_secret("{endpoint.id}") and update your receiver with the new secret.',
            err=as_json,
        )
    click.echo(f"Test it: mobilerun webhook ping {endpoint.id}", err=as_json)
    if receives_results:
        click.echo(
            f'Use it with @cloud("my-function", webhook="{endpoint.id}") or as endpointId in a flow\'s webhooks.send step.',
            err=as_json,
        )
    if as_json:
        click.echo(json.dumps(data, ensure_ascii=False))


@webhook.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def list_webhooks(as_json: bool) -> None:
    """List endpoints with destination query values masked."""
    endpoints = asyncio.run(CloudWebhooks().list())
    if as_json:
        click.echo(json.dumps([_metadata(e) for e in endpoints], ensure_ascii=False))
    elif not endpoints:
        render.empty("webhooks")
    else:
        render.heading(len(endpoints), "webhook")
        show_kind = any(e.kind != "http" for e in endpoints)
        table = Table(box=None, padding=(0, 1), pad_edge=False)
        for i, e in enumerate(endpoints):
            fields = _fields(e, show_kind)
            if i == 0:
                for key in fields:
                    table.add_column(key, overflow="fold")
            table.add_row(*fields.values())
        render.message(table)


def _fields(endpoint: WebhookEndpoint, show_kind: bool) -> dict[str, Text]:
    colors = {
        "ACTIVE": "green",
        "healthy": "green",
        "failing": "yellow",
        "degraded": "yellow",
        "DISABLED": "red",
        "blocked": "red",
    }
    fields = {
        "ID": Text(endpoint.id),
        "URL": Text(_endpoint_url(endpoint) or "—"),
        "Events": Text(", ".join(endpoint.event_types) or "all"),
        "State": Text(endpoint.state, style=colors.get(endpoint.state, "")),
        "Health": Text(endpoint.health, style=colors.get(endpoint.health, "")),
    }
    if show_kind:
        fields["Kind"] = Text(endpoint.kind)
    return fields


@webhook.command()
@click.argument("id", required=False)
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def get(id: str | None, as_json: bool) -> None:
    """Inspect a webhook with destination query values masked."""
    endpoint = asyncio.run(CloudWebhooks().get(_endpoint_id(id)))
    if as_json:
        click.echo(json.dumps(_metadata(endpoint), ensure_ascii=False))
    else:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="bold")
        grid.add_column(overflow="fold")
        for key, value in _fields(endpoint, endpoint.kind != "http").items():
            grid.add_row(key, value)
        render.message(Panel(grid, title="Webhook", expand=False))


@webhook.command()
@click.argument("id", required=False)
def ping(id: str | None) -> None:
    """Send a signed test event and report the receiver's HTTP result."""
    endpoint_id = _endpoint_id(id)
    success, status = asyncio.run(CloudWebhooks().ping(endpoint_id))
    click.echo(f"Ping sent to {endpoint_id}")
    if status is None:
        message = (
            "Delivered (no HTTP status reported)."
            if success
            else "Failed: no HTTP response from the receiver (possible network error or timeout)."
        )
    else:
        result = f"HTTP {status}"
        try:
            result += f" {HTTPStatus(status).phrase}"
        except ValueError:
            pass  # Nonstandard receiver status; keep the numeric code.
        message = (
            f"Delivered: {result}"
            if success
            else f"Failed: {result} (the receiver rejected the POST)"
        )
    click.echo(message)
    if not success:
        raise click.exceptions.Exit(1)


@webhook.command()
@click.argument("id", required=False)
@click.option("--yes", "-y", is_flag=True, help="Delete without prompting.")
def delete(id: str | None, yes: bool) -> None:
    """Delete the specified endpoint."""
    endpoint_id = _endpoint_id(id)
    if not yes:
        if not is_interactive():
            raise click.UsageError("Pass --yes to delete non-interactively.")
        if not confirm_prompt("Delete this webhook endpoint?", default=False):
            raise click.Abort()
    asyncio.run(CloudWebhooks().delete(endpoint_id))
    render.success("Webhook deleted.")
