"""Shared source selection for device transfer commands."""

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any

import click

from mobilerun.cli.base.menu import is_interactive
from mobilerun.cli.base.prompts import SelectChoice, select_prompt, text_prompt
from mobilerun.core.workspace import Workspace


class SourceCommand(click.Command):
    """Treat the lone positional argument as DEVICE when an ID is supplied."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        source_flag = "--app-id" if self.name == "install" else "--file-id"
        if any(arg == source_flag or arg.startswith(source_flag + "=") for arg in args):
            takes_value = {
                source_flag,
                "--dest",
                "--device",
                "-d",
                "--config",
                "--base-url",
            }
            positions: list[int] = []
            skip = False
            for index, arg in enumerate(args):
                if skip:
                    skip = False
                elif arg in takes_value:
                    skip = True
                elif not arg.startswith("-"):
                    positions.append(index)
            explicit_device = any(
                arg in {"--device", "-d"} or arg.startswith("--device=") for arg in args
            )
            if len(positions) == 1 and not explicit_device:
                index = positions[0]
                args = [*args[:index], "--device", args[index], *args[index + 1 :]]
        return super().parse_args(ctx, args)


def select_source(
    local: str | None, resource_id: str | None, *, apps: bool
) -> tuple[str | None, str | None]:
    """Require exactly one source, offering a local path or workspace picker in a TTY."""
    if local and resource_id:
        raise click.UsageError("Pass either a local path or a workspace ID, not both.")
    if not local and not resource_id and is_interactive():
        choice = select_prompt(
            "Choose a source",
            [
                SelectChoice("local", "Local file"),
                SelectChoice("workspace", "Workspace"),
            ],
        )
        if choice == "local":
            local = text_prompt("Local path").strip()
            if local:
                local = str(
                    click.Path(exists=True, dir_okay=False).convert(local, None, None)
                )
        elif choice == "workspace":
            if apps:
                choices = [
                    SelectChoice(item.id, item.name or item.package, item.id)
                    for item in asyncio.run(Workspace.list_apps())
                    if item.id is not None
                ]
            else:
                choices = [
                    SelectChoice(item.id, item.name, item.id)
                    for item in asyncio.run(Workspace.list_files())
                ]
            if not choices:
                raise click.ClickException("No workspace items found.")
            resource_id = select_prompt("Select a workspace item", choices)
        else:
            raise click.Abort()
    if not local and not resource_id:
        raise click.UsageError("Provide a local path or a workspace ID.")
    return local, resource_id


def source_option(
    local_name: str, id_name: str, *, apps: bool
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            kwargs[local_name], kwargs[id_name] = select_source(
                kwargs[local_name], kwargs[id_name], apps=apps
            )
            return func(*args, **kwargs)

        return wrapper

    return decorate
