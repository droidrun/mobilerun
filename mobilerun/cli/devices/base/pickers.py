"""Device-scoped choices, fetched only after opening the selected device."""

import click

from mobilerun.cli.base.menu import is_interactive
from mobilerun.cli.base.prompts import (
    SelectChoice,
    select_prompt,
    select_prompt_async,
    text_prompt,
    text_prompt_async,
)


def choose_value(value: str | None, label: str, choices: list[SelectChoice]) -> str:
    if value is not None:
        return value
    if not is_interactive():
        raise click.UsageError(f"Missing argument '{label.upper()}'.")
    selected = select_prompt(
        label, [*choices, SelectChoice("__manual__", "Enter manually…")]
    )
    if selected is None:
        raise click.exceptions.Exit(0)
    return text_prompt(label) if selected == "__manual__" else selected


async def choose_value_async(
    value: str | None, label: str, choices: list[SelectChoice]
) -> str:
    if value is not None:
        return value
    if not is_interactive():
        raise click.UsageError(f"Missing argument '{label.upper()}'.")
    selected = await select_prompt_async(
        label, [*choices, SelectChoice("__manual__", "Enter manually…")]
    )
    if selected is None:
        raise click.exceptions.Exit(0)
    return await text_prompt_async(label) if selected == "__manual__" else selected


async def choose_app(device, package: str | None) -> str:
    if package is not None or not is_interactive():
        return await choose_value_async(package, "package", [])
    try:
        apps = await device.list_apps()
    except Exception:
        click.echo("Could not load installed apps; enter a package manually.", err=True)
        return await choose_value_async(None, "package", [])
    choices = []
    for app in apps:
        key = app.get("package") or app.get("package_name") or app.get("packageName")
        if key:
            choices.append(SelectChoice(key, app.get("label") or key, key))
    return await choose_value_async(None, "package", choices)
