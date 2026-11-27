"""Interactive navigation and required-parameter prompts for Click commands."""

import sys
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from difflib import get_close_matches
from typing import Any

import click

from mobilerun.cli.base.errors import render_error
from mobilerun.cli.base.prompts import (
    SelectChoice,
    multi_select_prompt,
    select_prompt,
    text_prompt,
)
from mobilerun.utils.terminal import can_prompt


def is_interactive() -> bool:
    """Return whether a prompt drawn on stdout can be answered."""
    return can_prompt(sys.stdout)


ChoicesProvider = Callable[[click.Context], Sequence[SelectChoice]]


class PromptingParameter(click.Parameter):
    """Ask for a missing required value before Click validates it."""

    def __init__(
        self,
        *args: Any,
        choices: ChoicesProvider | None = None,
        prompt_label: str | None = None,
        empty_hint: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.choices = choices
        self.prompt_label = prompt_label
        self.empty_hint = empty_hint
        super().__init__(*args, **kwargs)

    def process_value(self, ctx: click.Context, value: Any) -> Any:
        if (
            self.value_is_missing(value)
            and self.required
            and not ctx.resilient_parsing
            and is_interactive()
        ):
            label = (
                self.prompt_label
                or (self.name or "value").replace("_", " ").capitalize()
            )
            try:
                choices = None
                if self.choices is not None:
                    try:
                        choices = self.choices(ctx)
                    except Exception:
                        click.echo(
                            "Warning: Could not load choices; enter a value manually.",
                            err=True,
                        )
                    else:
                        if not choices:
                            hint = f" {self.empty_hint}" if self.empty_hint else ""
                            click.echo(f"No {label.lower()} found.{hint}", err=True)
                            ctx.exit(0)
                elif isinstance(self.type, click.Choice):
                    choices = [
                        SelectChoice(str(item), str(item)) for item in self.type.choices
                    ]
                if choices is not None:
                    if self.nargs == -1:
                        with redirect_stdout(sys.stderr):
                            selected = multi_select_prompt(label, choices)
                        value = tuple(selected) if selected else None
                    else:
                        value = select_prompt(label, choices)
                elif self.nargs == -1:
                    entries = []
                    while True:
                        item = text_prompt(label).strip()
                        if not item:
                            if entries:
                                break
                            click.echo("Cancelled.", err=True)
                            ctx.exit(0)
                        entries.append(item)
                    value = tuple(entries)
                else:
                    value = text_prompt(label).strip()
                    if not value:
                        click.echo("Cancelled.", err=True)
                        ctx.exit(0)
            except (KeyboardInterrupt, click.Abort, EOFError):
                ctx.exit(130)
            if value is None:
                click.echo("Cancelled.", err=True)
                ctx.exit(0)
        return super().process_value(ctx, value)


class PromptingArgument(PromptingParameter, click.Argument):
    """An argument that prompts in interactive terminals."""


class PromptingOption(PromptingParameter, click.Option):
    """An option that prompts in interactive terminals."""


# Backwards compatibility for existing declarations.
ChoiceArgument = PromptingArgument
ChoiceOption = PromptingOption


def prepare_command(command: click.Command) -> click.Command:
    """Adapt ordinary required parameters, including pre-registered commands."""
    params: list[click.Parameter] = []
    for param in command.params:
        cls: type[PromptingArgument] | type[PromptingOption] | None = None
        if param.required and type(param) is click.Argument:
            cls = PromptingArgument
        elif param.required and type(param) is click.Option:
            cls = PromptingOption
        if cls is not None:
            adapted = cls.__new__(cls)
            adapted.__dict__.update(param.__dict__)
            adapted.choices = None
            adapted.prompt_label = None
            adapted.empty_hint = None
            params.append(adapted)
        else:
            params.append(param)
    command.params = params
    return command


class PromptingCommand(click.Command):
    def make_context(
        self,
        info_name: str | None,
        args: list[str],
        parent: click.Context | None = None,
        **extra: Any,
    ) -> click.Context:
        prepare_command(self)
        return super().make_context(info_name, args, parent=parent, **extra)


class _MenuBack(Exception):
    def __init__(self, path: list[str]) -> None:
        self.path = path


class MenuGroup(click.Group):
    command_class = PromptingCommand
    group_class = type

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        try:
            name, command, remaining = super().resolve_command(ctx, args)
        except click.UsageError as exc:
            matches = (
                get_close_matches(args[0], self.list_commands(ctx), n=1) if args else []
            )
            if matches and exc.message.startswith("No such command"):
                raise click.UsageError(
                    f"{exc.message} Did you mean '{matches[0]}'?", ctx
                ) from exc
            raise
        if command is not None:
            prepare_command(command)
        return name, command, remaining

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **kwargs: Any,
    ) -> Any:
        debug = "--debug" in (args if args is not None else sys.argv[1:])
        try:
            while True:
                try:
                    result = super().main(
                        args,
                        prog_name,
                        complete_var,
                        False,
                        windows_expand_args=windows_expand_args,
                        **kwargs,
                    )
                    break
                except _MenuBack as back:
                    args = (["--debug"] if debug else []) + back.path
        except (KeyboardInterrupt, click.Abort):
            if not standalone_mode:
                raise
            raise SystemExit(130) from None
        except click.ClickException as exc:
            if not standalone_mode:
                raise
            if exc.exit_code == 2:
                exc.show()
            else:
                render_error(exc, debug=debug)
            raise SystemExit(exc.exit_code) from None
        except Exception as exc:
            if not standalone_mode:
                raise
            render_error(exc, debug=debug)
            raise SystemExit(1) from None
        if not standalone_mode:
            return result
        raise SystemExit(result if isinstance(result, int) else 0)

    def menu_choices(self, ctx: click.Context) -> list[SelectChoice]:
        choices = []
        for name in self.list_commands(ctx):
            command = self.get_command(ctx, name)
            if command is not None and not command.hidden:
                choices.append(
                    SelectChoice(
                        name, name, command.short_help or command.get_short_help_str()
                    )
                )
        return choices

    def get_epilog(self, ctx: click.Context) -> str | None:
        return self.epilog

    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        epilog = self.get_epilog(ctx)
        if epilog:
            import inspect

            formatter.write_paragraph()
            with formatter.indentation():
                formatter.write_text(inspect.cleandoc(epilog))

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not args and not ctx.resilient_parsing and is_interactive():
            epilog = self.get_epilog(ctx)
            if epilog:
                click.echo(click.style(epilog, dim=True))
            choices = self.menu_choices(ctx)
            choices.append(SelectChoice("__back__", "Back" if ctx.parent else "Exit"))
            try:
                selected = select_prompt(
                    f"{ctx.command_path}: select a command", choices
                )
            except KeyboardInterrupt:
                ctx.exit(130)
            if selected is None or selected == "__back__":
                if ctx.parent is None:
                    ctx.exit(0)
                path: list[str] = []
                parent = ctx.parent
                while parent.parent is not None:
                    if parent.info_name is not None:
                        path.append(parent.info_name)
                    parent = parent.parent
                raise _MenuBack(list(reversed(path)))
            args = [selected]
        return super().parse_args(ctx, args)
