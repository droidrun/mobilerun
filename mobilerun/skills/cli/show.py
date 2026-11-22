import click

from mobilerun.cli.base.handling import handle_errors
from mobilerun.skills import list_skills
from mobilerun.skills.bundle import unknown_skill_message


@click.command("show")
@click.argument("name")
@handle_errors
def show_command(name: str) -> None:
    """Show a bundled skill's path and description."""
    skills = list_skills()
    skill = next((skill for skill in skills if skill.name == name), None)
    if skill is None:
        raise click.ClickException(
            unknown_skill_message(name, [skill.name for skill in skills])
        )
    click.echo(
        f"{skill.name}{f' [{skill.parent}]' if skill.parent else ''}\n"
        f"{skill.source}/{skill.relative_path}\n"
        f"{skill.source_file or skill.path / 'SKILL.md'}\n{skill.description}"
    )
