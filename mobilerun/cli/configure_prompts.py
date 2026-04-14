from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import click


@dataclass(frozen=True)
class SelectChoice:
    value: str
    label: str
    hint: str | None = None


def _import_inquirer_select():
    try:
        from InquirerPy import inquirer

        return inquirer
    except Exception:
        return None


def select_prompt(
    message: str,
    choices: Sequence[SelectChoice],
    *,
    default: str | None = None,
) -> str:
    """Interactive select prompt with arrow-key UX when available."""

    inquirer = _import_inquirer_select()
    if inquirer is not None:
        rendered = []
        for choice in choices:
            name = f"• {choice.label}"
            if choice.hint:
                name = f"{name} ({choice.hint})"
            rendered.append({"name": name, "value": choice.value})
        return str(
            inquirer.select(
                message=message,
                choices=rendered,
                default=default,
                vi_mode=False,
                cycle=False,
            ).execute()
        )

    click.echo(message)
    for index, choice in enumerate(choices, start=1):
        suffix = f" - {choice.hint}" if choice.hint else ""
        click.echo(f"  {index}. {choice.label}{suffix}")
    valid_values = [choice.value for choice in choices]
    return click.prompt(
        "Select option",
        type=click.Choice(valid_values, case_sensitive=False),
        default=default or valid_values[0],
        show_choices=True,
    )


def text_prompt(
    message: str,
    *,
    default: str | None = None,
    secret: bool = False,
) -> str:
    inquirer = _import_inquirer_select()
    normalized_default = default if default is not None else ""
    if inquirer is not None:
        prompt = inquirer.secret if secret else inquirer.text
        return str(prompt(message=message, default=normalized_default).execute()).strip()
    return click.prompt(message, default=normalized_default, hide_input=secret).strip()
