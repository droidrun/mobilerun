import json
import textwrap

import click

from mobilerun.cli.base import render
from mobilerun.cli.base.handling import handle_errors
from mobilerun.config.settings import get_settings
from mobilerun.skills import installed, list_skills


@click.command("list", short_help="List skills and installed targets")
@click.option("--long", "long_output", is_flag=True, help="Show full descriptions.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@handle_errors
def list_command(long_output: bool, as_json: bool) -> None:
    """List bundled skills and installed agent targets."""
    targets = {
        (agent, scope): installed(agent, scope)
        for agent, paths in get_settings().MOBILERUN_AGENTS.items()
        for scope in ("user", "project")
        if paths.skills and (scope == "user" or paths.skills[1] is not None)
    }
    skills = list_skills()
    entries = [
        {
            "name": skill.name,
            "description": skill.description,
            "source": skill.source,
            "path": str(skill.relative_path),
            "parent": skill.parent,
            "installed": [
                f"{agent}:{scope}"
                for (agent, scope), names in targets.items()
                if skill.name in names
            ],
        }
        for skill in skills
    ]
    if as_json:
        click.echo(json.dumps(entries, ensure_ascii=False))
    elif not entries:
        render.empty("skills")
    else:
        render.heading(len(entries), "skill")
        rows = []
        for entry in entries:
            description = entry["description"]
            if not long_output:
                description = textwrap.shorten(
                    description.split(". ", 1)[0], width=80, placeholder="…"
                )
            name = entry["name"]
            if entry["parent"]:
                name += f" [{entry['parent']}]"
            if entry["installed"]:
                name += f" ({', '.join(entry['installed'])})"
            rows.append([name, render.secondary(description)])
        render.table(rows)
