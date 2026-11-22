"""Manage the Mobilerun MCP connection for coding agents."""

import click

from mobilerun.cli.base.handling import handle_errors
from mobilerun.cli.base.menu import MenuGroup, is_interactive
from mobilerun.cli.base.prompts import SelectChoice, select_prompt
from mobilerun.config.settings import get_settings
from mobilerun.mcp import installer
from mobilerun.mcp.settings import get_mcp_settings


def _agent(agent: str | None) -> str | None:
    choices = [
        name for name, paths in get_settings().MOBILERUN_AGENTS.items() if paths.mcp
    ]
    if agent is not None:
        if agent not in choices:
            raise click.BadParameter(
                f"Choose from: {', '.join(choices)}", param_hint="--agent"
            )
        return agent
    if not is_interactive():
        raise click.MissingParameter(param_hint="--agent", param_type="option")
    try:
        return select_prompt(
            "Select agent", [SelectChoice(name, name) for name in choices]
        )
    except KeyboardInterrupt:
        raise click.exceptions.Exit(130) from None


@click.group(cls=MenuGroup)
def mcp() -> None:
    """Manage coding-agent MCP connections."""


@mcp.command("install")
@click.option("--agent", help="Coding agent.")
@click.option("--project", is_flag=True, help="Install for this project.")
@click.option("--force", is_flag=True, help="Replace an existing server.")
@click.option("--api-key", is_flag=True, help="Use MOBILERUN_API_KEY instead of OAuth.")
@handle_errors
def install_command(
    agent: str | None, project: bool, force: bool, api_key: bool
) -> None:
    """Install the public Mobilerun MCP server."""
    agent = _agent(agent)
    if agent is None:
        return
    scope = "project" if project else "user"
    result = installer.install(agent, scope, force=force, api_key=api_key)
    click.echo(f"{agent} ({scope}): {result}")
    if result in {"installed", "exists"}:
        if api_key:
            click.echo("Next: set MOBILERUN_API_KEY in the agent's environment")
        else:
            name = get_mcp_settings().MOBILERUN_MCP_SERVER_NAME
            fmt = get_settings().MOBILERUN_AGENTS[agent].mcp_format
            hint = fmt.oauth_sign_in if fmt else None
            click.echo(
                hint.replace("{server_name}", name)
                if hint
                else f"Next: sign in to the {name} MCP server in {agent}"
            )
        if agent == "codex" and project:
            click.echo("Codex project config requires trust.")


@mcp.command("uninstall")
@click.option("--agent", help="Coding agent.")
@click.option("--project", is_flag=True, help="Remove from this project.")
@handle_errors
def uninstall_command(agent: str | None, project: bool) -> None:
    """Remove the Mobilerun MCP server."""
    agent = _agent(agent)
    if agent is None:
        return
    scope = "project" if project else "user"
    click.echo(f"{agent} ({scope}): {installer.uninstall(agent, scope)}")


@mcp.command("list")
@handle_errors
def list_command() -> None:
    """Show installation status by agent and scope."""
    for agent, paths in get_settings().MOBILERUN_AGENTS.items():
        if paths.mcp:
            for scope, path in zip(("user", "project"), paths.mcp, strict=True):
                if path is not None:
                    click.echo(f"{agent} ({scope}): {installer.status(agent, scope)}")
