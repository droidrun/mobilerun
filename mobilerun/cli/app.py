"""Mobilerun CLI root with lazily loaded command groups."""

import logging
from importlib import import_module, metadata
from typing import cast

import click

from mobilerun import __version__
from mobilerun.addons.agents import agents_available
from mobilerun.cli.base.logging import cli_handler
from mobilerun.cli.base.menu import MenuGroup
from mobilerun.cli.base.prompts import SelectChoice

_COMMANDS = {
    "credentials": (
        "mobilerun.cli.credentials.group",
        "credentials",
        "Manage vault credentials for agent tasks.",
    ),
    "workspace": (
        "mobilerun.cli.workspace.group",
        "workspace",
        "Manage workspace files and apps.",
    ),
    "completion": (
        "mobilerun.cli.base.completion",
        "completion",
        "Install or print shell completion.",
    ),
    "proxy": ("mobilerun.cli.proxy.group", "proxy", "Manage Connect proxies."),
    "number": (
        "mobilerun.cli.phone_number.group",
        "number",
        "Manage cloud phone numbers.",
    ),
    "mailbox": (
        "mobilerun.cli.mailbox.group",
        "mailbox",
        "Manage cloud email mailboxes.",
    ),
    "mcp": ("mobilerun.mcp.cli.group", "mcp", "Manage coding-agent MCP connections."),
    "skills": ("mobilerun.skills.cli.group", "skills", "Manage reusable skills."),
    "account": ("mobilerun.cli.auth.group", "auth", "Manage cloud authentication."),
    "device": (
        "mobilerun.cli.devices.group",
        "device",
        "Interact with mobile devices.",
    ),
    "agent": (
        "mobilerun.agents.cli.agent.group",
        "agent",
        "Run an agent on a mobile device.",
    ),
    "macro": ("mobilerun.macro.cli.group", "macro", "Record and replay device macros."),
    "doctor": ("mobilerun.cli.doctor", "doctor", "Diagnose device connectivity."),
}

# Keep hidden commands addressable without importing command modules at startup.
_HIDDEN_COMMANDS = frozenset()

_ALIASES = {
    "run": ("mobilerun.agents.cli.agent.run", "run", "agent run"),
    "devices": ("mobilerun.cli.devices.list", "devices", "device list"),
    "login": ("mobilerun.cli.auth.login", "login", "account login"),
    "logout": ("mobilerun.cli.auth.logout", "logout", "account logout"),
    "whoami": ("mobilerun.cli.auth.status", "whoami", "account status"),
    "connect": ("mobilerun.cli.devices.connect", "connect", "device connect"),
    "disconnect": (
        "mobilerun.cli.devices.disconnect",
        "disconnect",
        "device disconnect",
    ),
    "setup": ("mobilerun.cli.devices.setup", "setup", "device setup"),
    "ping": ("mobilerun.cli.devices.ping", "ping", "device ping"),
    "configure": (
        "mobilerun.agents.cli.agent.llm.configure",
        "configure",
        "agent llm configure",
    ),
}


def _run_agent_macro(provider: str | None, model: str | None, **options) -> None:
    from mobilerun.addons.agents import require_agents

    with require_agents():
        from mobilerun.agents.agent_macro import AgentMacroOperator
        from mobilerun.agents.config.loader import ConfigLoader
        from mobilerun.agents.config.user_config import LLMProfile
    from mobilerun.macro.cli.replay import run_replay

    config = ConfigLoader.load(options["config_path"])
    if provider and model:
        config.llm_profiles = {
            name: LLMProfile(provider=provider, model=model)
            for name in config.llm_profiles
        }
    run_replay(AgentMacroOperator, config=config, **options)


class LazyGroup(MenuGroup):
    def menu_choices(self, ctx: click.Context) -> list[SelectChoice]:
        return [
            SelectChoice(name, name, _COMMANDS[name][2])
            for name in self.list_commands(ctx)
        ]

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(
            name
            for name in _COMMANDS
            if name not in _HIDDEN_COMMANDS and (name != "agent" or agents_available())
        )

    def get_epilog(self, ctx: click.Context) -> str | None:
        epilog = super().get_epilog(ctx)
        if not epilog or agents_available():
            return epilog
        return "; ".join(
            part
            for part in epilog.split("; ")
            if not part.startswith("mobilerun agent ")
        )

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        # Root help must not import optional command implementations.
        rows = [(name, _COMMANDS[name][2]) for name in self.list_commands(ctx)]
        with formatter.section("Commands"):
            formatter.write_dl(rows)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        from mobilerun.cli.compat import legacy_command

        if cmd_name in _ALIASES:
            return legacy_command(cmd_name, *_ALIASES[cmd_name])
        if cmd_name == "auth":
            cmd_name = "account"
        target = _COMMANDS.get(cmd_name)
        if target is None:
            return None
        module, attribute, _short_help = target
        if cmd_name == "macro":
            ctx.meta["macro_agent_handoff"] = _run_agent_macro
        # Commands are deliberately imported lazily from a static registry.
        return cast(click.Command, getattr(import_module(module), attribute))


def _print_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    try:
        version = metadata.version("mobilerun")
    except metadata.PackageNotFoundError:
        version = __version__
    click.echo(f"v{version}")
    ctx.exit()


@click.group(
    cls=LazyGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=(
        "Examples: mobilerun device list; mobilerun device (interactive menu); "
        'mobilerun agent run "open settings"; mobilerun skills install.'
    ),
)
@click.option("--debug", is_flag=True, help="Show full exception tracebacks.")
@click.option(
    "--version",
    is_flag=True,
    callback=_print_version,
    expose_value=False,
    is_eager=True,
    help="Show mobilerun version and exit",
)
def cli(debug: bool) -> None:
    """Mobilerun - Control your mobile devices through LLM agents."""
    logger = logging.getLogger("mobilerun")
    handler = cli_handler()
    from mobilerun.logs.handlers import configure_logging

    configure_logging(debug, handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
