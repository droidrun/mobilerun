"""Sandbox operations with explicit CLI authorization."""

import asyncio
import json
import shlex
import sys
from contextlib import nullcontext, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from mobilerun.cli.base import menu, render
from mobilerun.cli.base.billing import invoke_with_billing, require_consent_possible
from mobilerun.cli.base.errors import safe_error_message
from mobilerun.cli.base.prompts import SelectChoice, select_prompt_async
from mobilerun.cli.base.run import coro
from mobilerun.cli.base.safety import confirm_action_async
from mobilerun.errors.cloud import BillingRequired
from mobilerun.models.billing import Billing

if TYPE_CHECKING:
    from mobilerun.sandbox.models import DeploymentReceipt


class _TargetedBillingRequired(BillingRequired):
    def __init__(self, error: BillingRequired, target_summary: str) -> None:
        super().__init__(
            needed=error.needed, given=error.given, operation=error.operation
        )
        self.target_summary = target_summary

    @property
    def confirmation(self) -> str:
        return f"{self.target_summary}. {super().confirmation}"


_billing_option = click.option(
    "--billing",
    type=click.Choice([mode.value for mode in Billing]),
    help="Billing mode.",
)
_json_option = click.option("--json", "as_json", is_flag=True, help="Output JSON.")
_yes_option = click.option(
    "--yes", is_flag=True, help="Confirm the action without prompting."
)
_source_file = click.Path(exists=True, dir_okay=False, readable=True, path_type=Path)


class _StderrArgument(menu.PromptingArgument):
    """Keep required text prompts off stdout, including in JSON mode."""

    def process_value(self, ctx: click.Context, value: Any) -> Any:
        if (
            self.value_is_missing(value)
            and not ctx.resilient_parsing
            and menu.is_interactive()
        ):
            try:
                with redirect_stdout(sys.stderr):
                    value = menu.text_prompt(self.prompt_label).strip()
            except (KeyboardInterrupt, click.Abort, EOFError):
                ctx.exit(130)
            if not value:
                click.echo("Cancelled.", err=True)
                ctx.exit(0)
        return super().process_value(ctx, value)


def _resolve_mode(billing: str | None) -> Billing:
    from mobilerun.sandbox.models import resolve_billing

    return resolve_billing(billing)


def _deployment_choices(ctx: click.Context) -> list[SelectChoice]:
    from mobilerun.sandbox.flows import list_deployments

    mode = _resolve_mode(ctx.params.get("billing"))
    records = asyncio.run(list_deployments(billing=mode))
    choices = []
    for item in records:
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        latest = item.get("latestVersion", {})
        version = latest.get("version", "—")
        status = render.status(latest.get("status")).plain
        function_id = item["id"]
        short_id = f"{function_id[:8]}…" if len(function_id) > 8 else function_id
        choices.append(
            SelectChoice(name, f"{name} · {short_id} · {version} · {status}")
        )
    return choices


class _DeploymentArgument(menu.PromptingArgument):
    """Select deployed names without the generic argument's free-text fallback."""

    def __init__(
        self, *args: Any, requires_consent: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.requires_consent = requires_consent

    def process_value(self, ctx: click.Context, value: Any) -> Any:
        if (
            not self.value_is_missing(value)
            or ctx.resilient_parsing
            or not menu.is_interactive()
        ):
            return super().process_value(ctx, value)

        batch = self.nargs == -1
        if self.requires_consent:
            # Click processes explicitly supplied options before omitted arguments.
            # Refuse before the remote lookup when billing consent cannot be asked.
            require_consent_possible(yes=ctx.params.get("yes", False))
        choices = _deployment_choices(ctx)
        try:
            with redirect_stdout(sys.stderr):
                if not choices:
                    click.echo("No deployed sandbox functions found.")
                elif batch:
                    value = tuple(
                        menu.multi_select_prompt("Select sandbox functions", choices)
                        or ()
                    )
                else:
                    value = menu.select_prompt("Name", choices)
        except (KeyboardInterrupt, click.Abort, EOFError):
            ctx.exit(130)
        if not choices or not value:
            if choices:
                click.echo("Cancelled.", err=True)
            if batch and ctx.params.get("as_json"):
                click.echo("[]")
            ctx.exit(0)
        return super().process_value(ctx, value)


_name_argument = click.argument("name", cls=_DeploymentArgument)
_names_argument = click.argument(
    "name", cls=_DeploymentArgument, nargs=-1, required=True, metavar="NAME..."
)


async def _pick_device(*, as_json: bool, yes: bool) -> tuple[str | None, bool]:
    from mobilerun.cli.devices.base.discovery import discover_devices

    if not menu.is_interactive():
        raise click.UsageError(
            "Pass --device ID or --device-free for a scheduled deployment."
        )
    require_consent_possible(yes=yes)
    devices = await discover_devices(local=False, strict=True)
    # Disjoint choice values keep the device-free option distinct from device IDs.
    cloud = {f"device:{item.id}": item for item in devices if item.source == "cloud"}
    choices = [
        SelectChoice(
            value, item.name or item.id, f"{item.id} · {item.state or 'unknown'}"
        )
        for value, item in cloud.items()
    ]
    choices.append(SelectChoice("device-free", "Device-free"))
    with redirect_stdout(sys.stderr) if as_json else nullcontext():
        selected = await select_prompt_async("Choose a cloud device", choices)
    if selected is None:
        click.echo("Cancelled.", err=True)
        click.get_current_context().exit(0)
    return (None, True) if selected == "device-free" else (cloud[selected].id, False)


def _sample_value(schema: dict[str, Any]) -> Any:
    """Illustrate the shape without publishing literal defaults or examples."""
    for key in ("anyOf", "oneOf"):
        if schema.get(key):
            return _sample_value(schema[key][0])
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        return {
            key: _sample_value(properties.get(key, {}))
            for key in schema.get("required", [])
        }
    return (
        {"string": "", "integer": 0, "number": 0, "boolean": False, "array": []}.get(
            kind
        )
        if isinstance(kind, str)
        else None
    )


def _endpoint_contract(receipt: "DeploymentReceipt", schema: dict[str, Any]) -> None:
    endpoint = receipt.endpoint
    if endpoint is None:
        return
    base_url = endpoint.url.rsplit("/triggers/", 1)[0]
    sample = _sample_value(schema) or {}
    click.echo(
        f"Endpoint: POST {endpoint.url}\n"
        "Authorization: Bearer <YOUR_API_KEY> (use MOBILERUN_API_KEY)\n"
        "Idempotency-Key: <uuid> (optional)\n"
        "Content-Type: application/json\n"
        f"Example body: {json.dumps({'payload': sample})}\n"
        "HTTP 202 returns invocationId, enqueuedCount, and deduplicated, not the result.\n"
        f"Poll GET {base_url}/executions?flowId={receipt.flow_id}&invocationId=<invocationId>;\n"
        f"then GET {base_url}/executions/<executionId> for result.steps[].body.result.\n"
        f"List recent runs: mobilerun sandbox runs {shlex.quote(receipt.name)}"
    )


@click.command("deploy")
@click.argument("file", cls=_StderrArgument, prompt_label="File", type=_source_file)
@click.option("--device", metavar="ID", help="Use an existing cloud device.")
@click.option("--device-free", is_flag=True, help="Deploy without a device.")
@_billing_option
@_yes_option
@_json_option
@coro
async def deploy_command(
    file: Path,
    device: str | None,
    device_free: bool,
    billing: str | None,
    yes: bool,
    as_json: bool,
) -> None:
    """Deploy a cloud function as an endpoint by default."""
    from mobilerun.sandbox.billing import check_deploy_billing
    from mobilerun.sandbox.deployment import deploy_file
    from mobilerun.sandbox.packaging import package, read_declaration

    if device is not None and device_free:
        raise click.UsageError("Choose either --device or --device-free, not both.")
    if device is not None and not device.strip():
        raise click.UsageError("--device requires a nonempty ID.")
    try:
        source = file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise click.ClickException(f"Cannot read source: {file}") from exc
    declaration = read_declaration(file, source=source)
    if not device_free and device is None:
        device = declaration.device
        if device is None and declaration.schedule is not None:
            device, device_free = await _pick_device(as_json=as_json, yes=yes)

    target = f"device {device!r}" if device is not None else "device-free"
    summary = f"Deploy {declaration.name!r} ({target})"
    if declaration.schedule is not None:
        summary += f" on schedule {declaration.schedule!r}"

    async def authorize(attempt: Billing) -> None:
        try:
            await check_deploy_billing(attempt)
        except BillingRequired as exc:
            raise _TargetedBillingRequired(exc, summary) from exc

    _, mode = await invoke_with_billing(
        authorize, _resolve_mode(billing), yes=yes, confirm_paid=summary
    )
    artifact = package(file, source=source)

    receipt = await deploy_file(
        file,
        device=device,
        device_free=device_free,
        billing=mode,
        artifact=artifact,
    )
    if as_json:
        click.echo(json.dumps(receipt.to_dict()))
    else:
        render.success(f"{receipt.name} v{receipt.version}: {receipt.status}")
        fields = [("Function", receipt.function_id)]
        if receipt.flow_id:
            fields.append(("Flow", receipt.flow_id))
        render.kv(fields)
        _endpoint_contract(receipt, artifact.input_schema)
        for warning in receipt.warnings:
            render.warn(warning)


@click.command("list")
@_billing_option
@_json_option
@coro
async def list_command(billing: str | None, as_json: bool) -> None:
    """List deployed sandbox functions."""
    from mobilerun.sandbox.flows import list_deployments

    records = await list_deployments(billing=_resolve_mode(billing))
    if as_json:
        click.echo(json.dumps(records))
    elif not records:
        render.empty(
            "sandbox functions", "Run `mobilerun sandbox deploy FILE` to deploy one."
        )
    else:
        render.heading(len(records), "sandbox function")
        render.table(
            (
                item.get("name"),
                item["id"],
                latest.get("version"),
                render.status(latest.get("status")),
            )
            for item in records
            for latest in [item.get("latestVersion", {})]
        )


def _invalid_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _payload(input_json: str | None, input_file: Path | None) -> dict[str, Any]:
    if input_json is not None and input_file is not None:
        raise click.UsageError("Choose either --input or --input-file, not both.")
    if input_file is not None:
        try:
            input_json = input_file.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise click.UsageError("Cannot read --input-file as UTF-8.") from exc
    if input_json is None:
        return {}
    message = "Sandbox input must be a valid JSON object."
    try:
        payload = json.loads(input_json, parse_constant=_invalid_constant)
        if not isinstance(payload, dict):
            raise ValueError
        # Python also decodes overflowing exponents as infinity.
        json.dumps(payload, allow_nan=False)
    except (ValueError, RecursionError) as exc:
        raise click.UsageError(message) from exc
    return payload


@click.command("fire")
@click.argument("name", cls=_DeploymentArgument, requires_consent=True)
@click.option("--input", "input_json", metavar="JSON", help="Input as a JSON object.")
@click.option(
    "--input-file",
    type=_source_file,
    metavar="FILE",
    help="Read input from a JSON file.",
)
@_billing_option
@_yes_option
@_json_option
@coro
async def fire_command(
    name: str,
    input_json: str | None,
    input_file: Path | None,
    billing: str | None,
    yes: bool,
    as_json: bool,
) -> None:
    """Trigger a sandbox run without waiting for its result."""
    from mobilerun.sandbox.execution import fire_remote

    mode = _resolve_mode(billing)
    payload = _payload(input_json, input_file)
    result, _ = await invoke_with_billing(
        lambda attempt: fire_remote(name, payload, billing=attempt),
        mode,
        yes=yes,
        confirm_paid=f"Fire sandbox endpoint {name!r}",
    )
    invocation_id = result["invocation_id"]
    if as_json:
        click.echo(json.dumps({"invocation_id": invocation_id}))
    else:
        render.success(f"Fired invocation {invocation_id}")


def _run_summary(record: dict[str, Any]) -> dict[str, Any]:
    summary = {
        key: record[key]
        for key in ("id", "status", "startedAt", "finishedAt", "invocationId", "flowId")
        if key in record and (record[key] is None or isinstance(record[key], str))
    }
    start, end = summary.get("startedAt"), summary.get("finishedAt")
    if start and end:
        try:
            summary["duration"] = (
                datetime.fromisoformat(end) - datetime.fromisoformat(start)
            ).total_seconds()
        except (ValueError, TypeError):
            pass
    return summary


@click.command("runs")
@_name_argument
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Maximum runs.",
)
@_billing_option
@_json_option
@coro
async def runs_command(
    name: str, limit: int, billing: str | None, as_json: bool
) -> None:
    """List recent runs of a sandbox function."""
    from mobilerun.sandbox.execution import recent_runs

    records = await recent_runs(name, limit, billing=_resolve_mode(billing))
    if as_json:
        click.echo(json.dumps([_run_summary(record) for record in records]))
    elif not records:
        render.empty("sandbox runs")
    else:
        render.heading(len(records), "sandbox run")
        render.table(
            (
                item["id"],
                render.status(item.get("status")),
                render.relative_time(item.get("startedAt")),
            )
            for item in records
        )


async def _set_enabled(
    names: tuple[str, ...], enabled: bool, billing: str | None, yes: bool, as_json: bool
) -> None:
    from mobilerun.sandbox.billing import require_paid
    from mobilerun.sandbox.flows import set_enabled

    mode = _resolve_mode(billing)
    verb = "Enable" if enabled else "Disable"
    names = tuple(dict.fromkeys(names))
    targets = ", ".join(map(repr, names))
    warning = f"{verb} {targets}. Billing: {mode}."
    if enabled:
        summary = f"Enable sandbox flows {targets}"

        async def authorize(attempt: Billing) -> None:
            try:
                require_paid(billing=attempt)
            except BillingRequired as exc:
                raise _TargetedBillingRequired(exc, summary) from exc

        _, mode = await invoke_with_billing(
            authorize, mode, yes=yes, confirm_paid=summary
        )
    else:
        try:
            await confirm_action_async(
                "disable sandbox flows", mode, warning, yes=yes, err=True
            )
        except click.exceptions.Exit as exc:
            if as_json and exc.exit_code == 0:
                click.echo("[]")
            raise

    results = []
    for name in names:
        try:
            await set_enabled(name, enabled, billing=mode)
        except (BillingRequired, click.Abort, click.exceptions.Exit):
            raise
        except Exception as exc:
            status = "failed"
            render.error(f"{name!r}: {safe_error_message(exc)}")
        else:
            status = "enabled" if enabled else "disabled"
            if not as_json:
                render.success(f"{name!r}: {status}")
        results.append({"name": name, "status": status})
    if as_json:
        click.echo(json.dumps(results))
    if any(item["status"] == "failed" for item in results):
        click.get_current_context().exit(1)


@click.command("enable")
@click.argument(
    "name",
    cls=_DeploymentArgument,
    nargs=-1,
    required=True,
    metavar="NAME...",
    requires_consent=True,
)
@_billing_option
@_yes_option
@_json_option
@coro
async def enable_command(
    name: tuple[str, ...], billing: str | None, yes: bool, as_json: bool
) -> None:
    """Enable one or more owned sandbox flows."""
    await _set_enabled(name, True, billing, yes, as_json)


@click.command("disable")
@_names_argument
@_billing_option
@_yes_option
@_json_option
@coro
async def disable_command(
    name: tuple[str, ...], billing: str | None, yes: bool, as_json: bool
) -> None:
    """Disable one or more owned sandbox flows."""
    await _set_enabled(name, False, billing, yes, as_json)
