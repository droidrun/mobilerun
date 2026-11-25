"""Inspect plugin metadata without loading implementations by default."""

import json

import click

from mobilerun.addons import plugins as registry
from mobilerun.addons.settings import CLI_ENTRY_POINT_GROUP, OPERATOR_ENTRY_POINT_GROUP
from mobilerun.cli.base.menu import MenuGroup


@click.group(cls=MenuGroup)
def plugins() -> None:
    """Inspect installed Mobilerun plugins."""


@plugins.command("list")
@click.option("--check", is_flag=True, help="Load plugins to check their status.")
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def list_plugins(check: bool, as_json: bool) -> None:
    """List installed plugin entry points and their distributions."""
    from mobilerun.cli.app import reserved_command_names

    reserved = reserved_command_names()
    rows = []
    for group in (CLI_ENTRY_POINT_GROUP, OPERATOR_ENTRY_POINT_GROUP):
        entries = sorted(
            registry.installed_entries(group),
            key=lambda ep: (ep.name, str(ep.value)),
        )
        selected = registry.discover(group, entries=entries)
        for entry in entries:
            status = "not checked"
            if selected.get(entry.name) is not entry or (
                group == CLI_ENTRY_POINT_GROUP and entry.name in reserved
            ):
                status = "shadowed"
            elif check:
                try:
                    value = entry.load()
                    valid = (
                        isinstance(value, click.Command)
                        if group == CLI_ENTRY_POINT_GROUP
                        else callable(value)
                    )
                    status = "ok" if valid else "broken"
                except Exception:
                    status = "broken"
            dist = entry.dist
            rows.append(
                {
                    "distribution": (registry.distribution_name(entry)),
                    "version": dist.version if dist else "unknown",
                    "group": group,
                    "name": entry.name,
                    "status": status,
                }
            )
    if as_json:
        click.echo(json.dumps(rows))
    elif not rows:
        click.echo("No plugins installed.")
    else:
        for row in rows:
            click.echo(
                f"{row['distribution']} {row['version']}  {row['group']}  {row['name']}  {row['status']}"
            )
