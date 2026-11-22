"""Shared skill command options and errors."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click

from mobilerun.cli.base.menu import ChoiceOption
from mobilerun.config.settings import get_settings

F = TypeVar("F", bound=Callable[..., Any])


def _skill_agents() -> list[str]:
    return [
        name for name, paths in get_settings().MOBILERUN_AGENTS.items() if paths.skills
    ]


class _AgentOption(ChoiceOption):
    """Resolve allowed agents when the command is invoked or help is rendered."""

    def process_value(self, ctx: click.Context, value: Any) -> Any:
        self.type = click.Choice(_skill_agents())
        return super().process_value(ctx, value)

    def shell_complete(self, ctx: click.Context, incomplete: str):
        self.type = click.Choice(_skill_agents())
        return super().shell_complete(ctx, incomplete)

    def get_help_record(self, ctx: click.Context) -> tuple[str, str] | None:
        self.type = click.Choice(_skill_agents())
        return super().get_help_record(ctx)


def options(function: F) -> F:
    function = (
        click.option("--force", is_flag=True, help="Replace existing skills.")(function)
        if function.__name__ == "install_command"
        else function
    )
    function = click.option(
        "--dir",
        "directory",
        type=click.Path(path_type=Path),
        help="Install directory override.",
    )(function)
    function = click.option(
        "--project",
        is_flag=True,
        help="Install in the current project's skills directory.",
    )(function)
    return click.option(
        "--agent",
        cls=_AgentOption,
        type=click.Choice(()),
        required=function.__name__ != "install_command",
    )(function)
