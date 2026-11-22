"""Shared interactive prompts."""

import asyncio
from dataclasses import dataclass
from typing import Sequence

import click
from InquirerPy import inquirer  # pyright: ignore[reportPrivateImportUsage]
from InquirerPy.base.simple import BaseSimplePrompt
from prompt_toolkit.key_binding import KeyPressEvent


@dataclass(frozen=True)
class SelectChoice:
    value: str
    label: str
    hint: str | None = None


def _execute(prompt):
    """Execute a prompt from a synchronous Click parsing context."""
    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return prompt.execute()
        raise click.ClickException(
            "Interactive prompt failed: a synchronous prompt cannot run inside an event loop. "
            "Use confirm_action_async() or the corresponding *_async prompt "
            "in async commands."
        )
    except (click.ClickException, click.Abort, EOFError, KeyboardInterrupt):
        raise
    except Exception as exc:
        raise click.ClickException(
            "Interactive prompt failed; use --debug for details."
        ) from exc


async def _execute_async(prompt):
    """Keep input and cancellation on the command's running event loop."""
    try:
        return await prompt.execute_async()
    except (asyncio.CancelledError, KeyboardInterrupt):
        application = (
            prompt._application
            if hasattr(prompt, "_application")
            else prompt._session.app
        )
        if application.is_running:
            application.exit()
        raise
    except (click.ClickException, click.Abort, EOFError):
        raise
    except Exception as exc:
        raise click.ClickException(
            "Interactive prompt failed; use --debug for details."
        ) from exc


def _select(
    message: str, choices: Sequence[SelectChoice], *, default: str | None = None
) -> BaseSimplePrompt:
    rendered = [
        {
            "name": f"• {choice.label}" + (f" ({choice.hint})" if choice.hint else ""),
            "value": choice.value,
        }
        for choice in choices
    ]
    prompt = inquirer.select(  # pyright: ignore[reportPrivateImportUsage]
        message=message, choices=rendered, default=default, vi_mode=False, cycle=False
    )

    @prompt.register_kb("escape")
    def cancel(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    return prompt


def select_prompt(
    message: str, choices: Sequence[SelectChoice], *, default: str | None = None
) -> str | None:
    return _execute(_select(message, choices, default=default))


async def select_prompt_async(
    message: str, choices: Sequence[SelectChoice], *, default: str | None = None
) -> str | None:
    return await _execute_async(_select(message, choices, default=default))


def confirm_prompt(message: str, *, default: bool = False) -> bool:
    """Ask for affirmative consent; Escape is always a refusal."""
    return (
        select_prompt(
            message,
            [SelectChoice("yes", "Yes"), SelectChoice("no", "No")],
            default="yes" if default else "no",
        )
        == "yes"
    )


async def confirm_prompt_async(message: str, *, default: bool = False) -> bool:
    return (
        await select_prompt_async(
            message,
            [SelectChoice("yes", "Yes"), SelectChoice("no", "No")],
            default="yes" if default else "no",
        )
        == "yes"
    )


def multi_select_prompt(
    message: str, choices: Sequence[SelectChoice]
) -> list[str] | None:
    rendered = [
        {
            "name": choice.label + (f" ({choice.hint})" if choice.hint else ""),
            "value": choice.value,
        }
        for choice in choices
    ]
    prompt = inquirer.checkbox(message=message, choices=rendered, vi_mode=False)

    @prompt.register_kb("escape")
    def cancel(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    return _execute(prompt)


def _text(
    message: str, *, default: str | None = None, secret: bool = False
) -> BaseSimplePrompt:
    prompt = (
        inquirer.secret  # pyright: ignore[reportPrivateImportUsage]
        if secret
        else inquirer.text  # pyright: ignore[reportPrivateImportUsage]
    )
    instance = prompt(message=message, default=default or "")

    @instance.register_kb("escape")
    def cancel(event: KeyPressEvent) -> None:
        event.app.exit(result="")

    return instance


def text_prompt(message, *, default=None, secret=False) -> str:
    return str(_execute(_text(message, default=default, secret=secret))).strip()


async def text_prompt_async(message, *, default=None, secret=False) -> str:
    return str(
        await _execute_async(_text(message, default=default, secret=secret))
    ).strip()
