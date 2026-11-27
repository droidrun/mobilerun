"""Replay a recorded macro strictly, without agent dependencies."""

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any, TypedDict, TypeVar, Unpack

import click

from mobilerun.cli.base import render
from mobilerun.cli.base.billing import confirm_paid_action
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.config.loader import ConfigLoader
from mobilerun.config.settings import get_settings
from mobilerun.core.credentials import DictCredentials, FileCredentials
from mobilerun.macro import MacroOperator, MacroStep, load_macro, resolve_macro_path
from mobilerun.macro.builder import _replay_snapshot
from mobilerun.macro.settings import get_macro_settings
from mobilerun.macro.utils import configure_macro_logging
from mobilerun.models.results import ExecutionResult

F = TypeVar("F", bound=Callable[..., Any])


class ReplayOptions(TypedDict):
    path: str
    target: DeviceTarget
    delay: float
    start_from: int
    max_steps: int | None
    debug: bool
    dry_run: bool
    state_timeout: float
    state_threshold: float
    config_path: str | None


def replay_options(command: F) -> F:
    """Shared replay controls for strict and agent-macro commands."""
    options = [
        click.argument("path", metavar="MACRO"),
        click.option(
            "--delay",
            "-t",
            default=lambda: get_macro_settings().MOBILERUN_MACRO_DELAY,
            type=click.FloatRange(min=0),
            help="Delay between actions (seconds)",
        ),
        click.option(
            "--start-from",
            "-s",
            default=1,
            type=click.IntRange(min=1),
            help="Start from step (1-based)",
        ),
        click.option("--max-steps", "-m", default=None, type=click.IntRange(min=0)),
        click.option(
            "--state-timeout",
            default=lambda: get_macro_settings().MOBILERUN_MACRO_STATE_TIMEOUT,
            type=click.FloatRange(min=0),
            show_default=True,
        ),
        click.option(
            "--state-threshold",
            default=lambda: get_macro_settings().MOBILERUN_MACRO_STATE_THRESHOLD,
            type=click.FloatRange(min=0, max=1),
            show_default=True,
        ),
        click.option("--debug", is_flag=True),
        click.option(
            "--dry-run", is_flag=True, help="Show actions without connecting a device"
        ),
        click.option(
            "--config", "config_path", default=None, help="Config file for credentials"
        ),
        device_command(allow_auto_setup=True),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def _selected_actions(
    steps: list[MacroStep], start_from: int, max_steps: int | None
) -> list[MacroStep]:
    """Read the same step range as the player, with a one-based CLI start."""
    end = None if max_steps is None else start_from - 1 + max_steps
    return steps[start_from - 1 : end]


def _preflight_before_target(command: Callable[..., None]) -> Callable[..., None]:
    """Check the recording before the device-command wrapper can discover a target."""

    @wraps(command)
    def invoke(*, yes: bool, **options: Any) -> None:
        path = resolve_macro_path(options["path"])
        steps = load_macro(path)
        if not (options["dry_run"] or yes):
            count = 0
            for step in _selected_actions(
                steps, options["start_from"], options["max_steps"]
            ):
                if step.action != "extension_call":
                    continue
                name = step.args.get("name")
                # Extension roles are not recorded; a named .send may be SMS.
                if isinstance(name, str):
                    owner, _, method = name.rpartition(".")
                    if owner and method == "send":
                        count += 1
            if count:
                asyncio.run(confirm_paid_action(f"Send {count} SMS", yes=False))

        # Play what was checked, even if the file changes while the device opens.
        options["path"] = str(path)
        token = _replay_snapshot.set((path, steps))
        try:
            return command(**options)
        finally:
            _replay_snapshot.reset(token)

    return invoke


@click.command(short_help="Replay strictly")
@click.option("--on-mismatch", type=click.Choice(["stop", "agent"]), default="stop")
@click.option("--provider", default=None)
@click.option("--model", default=None)
@click.option(
    "--yes", "-y", is_flag=True, help="Confirm possible SMS charges without prompting."
)
@_preflight_before_target
@replay_options
def replay(
    on_mismatch: str,
    provider: str | None,
    model: str | None,
    **options: Unpack[ReplayOptions],
) -> None:
    """Replay a macro name, folder, or pinned version file on DEVICE."""
    if bool(provider) != bool(model):
        raise click.UsageError("Provide both --provider and --model, or neither")
    if on_mismatch == "agent" and not options["dry_run"]:
        handoff = click.get_current_context().meta.get("macro_agent_handoff")
        if handoff is None:
            raise click.UsageError(
                "Agent handoff requires the 'mobilerun macro replay' command"
            )
        handoff(provider=provider, model=model, **options)
    else:
        run_replay(MacroOperator, **options)


def _action_summary(step: MacroStep) -> str:
    """Preview the operation without displaying snapshots or free-form payloads."""
    kind = step.action
    if kind in {"type_secret", "input_text", "extension_call", "open_url"}:
        return f"{kind} [redacted]"
    fields = {
        "tap": ("x", "y"),
        "swipe": ("start_x", "start_y", "end_x", "end_y", "duration_ms"),
        "drag": ("start_x", "start_y", "end_x", "end_y", "duration", "duration_ms"),
        "wait": ("duration",),
        "press_key_code": ("key_code",),
    }
    if kind not in fields and kind not in {"start_app", "stop_app", "button_press"}:
        return "unknown"
    parts = [kind]
    for key in fields.get(kind, ()):
        value = step.args.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key}={str(value)[:24]}")
    return " ".join(parts)


def run_replay(
    operator_type: Callable[..., MacroOperator],
    *,
    path: str,
    target: DeviceTarget,
    delay: float,
    start_from: int,
    max_steps: int | None,
    debug: bool,
    dry_run: bool,
    state_timeout: float,
    state_threshold: float,
    config_path: str | None,
    **operator_options: object,
) -> None:
    configure_macro_logging(debug)
    snapshot = _replay_snapshot.get()
    if snapshot is not None and str(snapshot[0]) == path:
        resolved, steps = snapshot
    else:
        resolved = resolve_macro_path(path)
        steps = load_macro(resolved)
    if dry_run:
        for step in _selected_actions(steps, start_from, max_steps):
            click.echo(f"{step.i}: {_action_summary(step)}")
        return
    config = ConfigLoader.load(config_path)
    operator = operator_type(
        start_from_step=start_from - 1,
        delay_between_actions=delay,
        state_timeout=state_timeout,
        state_threshold=state_threshold,
        **operator_options,
    )

    async def execute() -> ExecutionResult[Any]:
        async with target.open() as device:
            configured = config.credentials
            if configured.get("enabled"):
                await FileCredentials(
                    configured.get(
                        "file_path", str(get_settings().local_credentials_path)
                    )
                ).bind(device)
            elif configured and "enabled" not in configured:
                await DictCredentials(configured).bind(device)
            return await operator.execute(device, str(resolved), max_steps=max_steps)

    token = _replay_snapshot.set((resolved, steps))
    try:
        result = asyncio.run(execute())
    finally:
        _replay_snapshot.reset(token)
    if not result.success:
        raise click.ClickException(result.answer or "Macro replay failed")
    render.success("Replay completed")
