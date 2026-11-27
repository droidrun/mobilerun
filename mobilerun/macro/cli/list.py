"""List recorded macros."""

import json

import click

from mobilerun.cli.base import render
from mobilerun.macro import load_macro, resolve_macro_path
from mobilerun.macro.errors import MacroError
from mobilerun.macro.settings import get_macro_settings
from mobilerun.utils.paths import PathResolver, local_store_path


@click.command("list")
@click.argument(
    "directory",
    type=click.Path(),
    default=lambda: local_store_path(
        get_macro_settings().macro_directory,
        overridden=False,
    ).path,
)
@click.option("--debug", is_flag=True, help="Enable debug logging", default=False)
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def list(directory: str, debug: bool, as_json: bool) -> None:
    """List the latest version of each macro in a directory."""
    resolved_dir = PathResolver.resolve(directory)
    entries = []
    if resolved_dir.is_dir():
        for item in sorted(resolved_dir.iterdir()):
            if not item.is_dir():
                continue
            try:
                path = resolve_macro_path(item)
            except MacroError:
                continue
            try:
                entry = {
                    "name": item.name,
                    "version": path.name,
                    "actions": len(load_macro(path)),
                }
            except MacroError:
                entry = {
                    "name": item.name,
                    "version": path.name,
                    "actions": 0,
                    "error": "Error loading",
                }
            entries.append(entry)
    elif resolved_dir.exists():
        raise click.ClickException(f"Not a directory: {directory}")
    elif (
        click.get_current_context().get_parameter_source("directory")
        == click.core.ParameterSource.COMMANDLINE
    ):
        raise click.ClickException(f"Macro directory not found: {directory}")
    if as_json:
        click.echo(json.dumps(entries, ensure_ascii=False))
    elif not entries:
        render.empty("macros")
    else:
        render.heading(len(entries), "macro")
        rows = []
        for entry in entries:
            rows.append(
                [
                    entry["name"],
                    render.secondary(entry["version"]),
                    render.secondary(entry.get("error", f"{entry['actions']} actions")),
                ]
            )
        render.table(rows)
