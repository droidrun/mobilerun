from pathlib import Path

import click
from InquirerPy.prompts.fuzzy import FuzzyPrompt
from prompt_toolkit.key_binding import KeyPressEvent

from mobilerun.cli.base.handling import handle_errors
from mobilerun.cli.base.menu import is_interactive
from mobilerun.cli.base.prompts import SelectChoice, select_prompt
from mobilerun.config.settings import get_settings
from mobilerun.skills import install, list_skills
from mobilerun.skills.bundle import Skill, bundle_folders, install_bundles

from .common import options

_EVERYTHING = "/"  # Not a valid skill name.


def select_skills(
    skills: list[Skill], *, include_all: bool = False
) -> list[str] | None:
    choices = [
        {
            "name": f"• {skill.name} ({skill.description[:100]})"
            + (f" [{skill.parent}]" if skill.parent else "")
            + (f" [{skill.source}/{skill.relative_path}]" if skill.source else ""),
            "value": skill.name,
        }
        for skill in skills
    ]
    if include_all:
        choices.insert(0, {"name": "Everything (all skills)", "value": _EVERYTHING})
    names = {choice["name"]: choice["value"] for choice in choices}
    names["Everything (all skills)"] = "Everything (all skills)"
    prompt = FuzzyPrompt(
        message="Select skills",
        choices=choices,
        transformer=lambda selected: ", ".join(names[name] for name in selected),
        vi_mode=False,
        cycle=False,
        multiselect=True,
        instruction="Type to filter; use Tab to select, Enter to install.",
    )

    @prompt.register_kb("escape")
    def cancel(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    return prompt.execute()


@click.command("install", short_help="Install skills for an agent")
@click.argument("names", nargs=-1)
@click.option("--all", "all_skills", is_flag=True, help="Install all bundled skills.")
@options
@handle_errors
def install_command(
    names: tuple[str, ...],
    all_skills: bool,
    agent: str | None,
    project: bool,
    directory: Path | None,
    force: bool,
) -> None:
    """Install named skills, --all, or choose skills interactively."""
    mode = "legacy"
    if names and names[0] in ("bundle", "single"):
        mode, names = names[0], names[1:]
    if names and all_skills:
        raise click.UsageError("Specify skill names or --all, not both.")
    if not names and not all_skills and not is_interactive():
        raise click.UsageError("Specify skill names or --all.")
    flat_picker = False
    if mode == "legacy" and not names:
        if all_skills:
            mode = "bundle"
        elif all(len(skill.relative_path.parts) == 1 for skill in list_skills()):
            flat_picker = True
        else:
            try:
                mode = select_prompt(
                    "Select install mode",
                    [
                        SelectChoice("everything", "Everything"),
                        SelectChoice("bundle", "Bundle (pick folders)"),
                        SelectChoice("single", "Single (pick skills)"),
                    ],
                )
            except KeyboardInterrupt:
                raise click.exceptions.Exit(130) from None
            if mode is None:
                return
            if mode == "everything":
                mode, all_skills = "bundle", True
    if agent is None:
        if not is_interactive():
            raise click.MissingParameter(param_hint="--agent", param_type="option")
        try:
            agent = select_prompt(
                "Select agent",
                [
                    SelectChoice(name, name)
                    for name, paths in get_settings().MOBILERUN_AGENTS.items()
                    if paths.skills
                ],
            )
        except KeyboardInterrupt:
            raise click.exceptions.Exit(130) from None
        if agent is None:
            return
    if mode == "bundle":
        available = bundle_folders()
        if all_skills:
            selected = list(available)
        elif names:
            selected = list(names)
        else:
            try:
                skills = list_skills()
                selected = select_skills(
                    [
                        Skill(
                            name,
                            f"{sum(s.source == path.parent.name and s.relative_path.parts[0] == name for s in skills)} skills, {path.parent.name}",
                            path,
                        )
                        for name, path in available.items()
                    ]
                )
            except KeyboardInterrupt:
                raise click.exceptions.Exit(130) from None
        installer = install_bundles
    else:
        if all_skills:
            selected = [skill.name for skill in list_skills() if not skill.parent]
        elif names:
            selected = list(names)
        else:
            try:
                selected = (
                    select_skills(list_skills(), include_all=True)
                    if flat_picker
                    else select_skills(list_skills())
                )
            except KeyboardInterrupt:
                raise click.exceptions.Exit(130) from None
        installer = install
    if flat_picker and selected is not None and _EVERYTHING in selected:
        selected = list(bundle_folders())
        installer = install_bundles
    if selected is None:
        return
    if not selected:
        click.echo("No skills selected.")
        return
    for result in installer(
        selected, agent, "project" if project else "user", directory, force
    ):
        click.echo(f"{result.name}: {result.status} ({result.path})")
