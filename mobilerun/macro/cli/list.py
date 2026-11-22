"""List recorded macros."""

import json

import click

from mobilerun.cli.base import render
from mobilerun.macro import MacroPlayer
from mobilerun.macro.settings import get_macro_settings
from mobilerun.config.paths import PathResolver


@click.command("list")
@click.argument(
    "directory",
    type=click.Path(),
    default=lambda: get_macro_settings().MOBILERUN_MACRO_DIRECTORY,
)
@click.option("--debug", is_flag=True, help="Enable debug logging", default=False)
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def list(directory: str, debug: bool, as_json: bool) -> None:
    """List available trajectory folders in a directory."""
    resolved_dir = PathResolver.resolve(directory)
    entries = []
    if resolved_dir.is_dir():
        for item in sorted(resolved_dir.iterdir()):
            if item.is_dir() and (item / "macro.json").exists():
                try:
                    data = MacroPlayer().load_macro_from_folder(str(item))
                    entries.append(
                        {
                            "name": item.name,
                            "description": data.get("description", "No description"),
                            "actions": data.get("total_actions", 0),
                        }
                    )
                except Exception:
                    entries.append(
                        {
                            "name": item.name,
                            "description": "Error loading",
                            "actions": 0,
                        }
                    )
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
                    render.secondary(entry["description"]),
                    render.secondary(f"{entry['actions']} actions"),
                ]
            )
        render.table(rows)
