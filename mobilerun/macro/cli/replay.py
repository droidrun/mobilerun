"""Replay a recorded macro strictly, without agent dependencies."""

import asyncio
from collections.abc import Callable
from typing import Any, TypedDict, TypeVar, Unpack

import click

from mobilerun.cli.base import render
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.config.loader import ConfigLoader
from mobilerun.config.settings import get_settings
from mobilerun.core.credentials import DictCredentials, FileCredentials
from mobilerun.macro import MacroOperator, MacroPlayer
from mobilerun.macro.settings import get_macro_settings
from mobilerun.macro.utils import configure_macro_logging
from mobilerun.models.results import ExecutionResult
from mobilerun.utils.paths import PathResolver

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
        click.argument("path", type=click.Path(exists=True)),
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


@click.command(short_help="Replay strictly")
@click.option("--on-mismatch", type=click.Choice(["stop", "agent"]), default="stop")
@click.option("--provider", default=None)
@click.option("--model", default=None)
@replay_options
def replay(
    on_mismatch: str,
    provider: str | None,
    model: str | None,
    **options: Unpack[ReplayOptions],
) -> None:
    """Replay FILE (or trajectory folder) on DEVICE; stop on state mismatch."""
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


def _action_summary(action: dict[str, Any]) -> str:
    """Preview the operation without displaying snapshots or free-form payloads."""
    kind = action.get("action_type", action.get("type"))
    if (
        not isinstance(kind, str)
        or len(kind) > 32
        or not kind.isascii()
        or not kind.isidentifier()
    ):
        return "unknown"
    if kind in {"type_secret", "input_text", "extension_call", "open_url"}:
        return f"{kind} [redacted]"
    fields = {
        "tap": ("x", "y"),
        "swipe": ("start_x", "start_y", "end_x", "end_y", "duration_ms"),
        "drag": ("start_x", "start_y", "end_x", "end_y", "duration", "duration_ms"),
        "wait": ("duration",),
        "key_press": ("keycode",),
        "press_key_code": ("key_code",),
    }
    parts = [kind]
    for key in fields.get(kind, ()):
        value = action.get(key)
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
    try:
        resolved = str(PathResolver.resolve(path, must_exist=True))
        data = MacroPlayer().load_macro_from_file(resolved)
        if dry_run:
            actions = data.get("actions", [])[start_from - 1 :]
            if max_steps is not None:
                actions = actions[:max_steps]
            for index, action in enumerate(actions, start=start_from):
                click.echo(f"{index}: {_action_summary(action)}")
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
                return await operator.execute(device, resolved, max_steps=max_steps)

        result = asyncio.run(execute())
        if not result.success:
            raise click.ClickException(result.answer or "Macro replay failed")
        render.success("Replay completed")
    except click.ClickException:
        raise
