"""Mobilerun CLI root with lazily loaded command groups."""

import logging
from importlib import import_module, metadata
from typing import Any, cast

import click
from click.shell_completion import CompletionItem

from mobilerun import __version__
from mobilerun.addons import plugins
from mobilerun.addons.settings import (
    CLI_ENTRY_POINT_GROUP,
    CLI_PLUGIN_HELP,
    OPERATOR_ENTRY_POINT_GROUP,
)
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
    "account": ("mobilerun.cli.auth.group", "auth", "Manage cloud authentication."),
    "device": (
        "mobilerun.cli.devices.group",
        "device",
        "Interact with mobile devices.",
    ),
    "doctor": ("mobilerun.cli.doctor", "doctor", "Diagnose device connectivity."),
    "plugins": ("mobilerun.cli.plugins.group", "plugins", "Inspect installed plugins."),
}

# Keep hidden commands addressable without importing command modules at startup.
_HIDDEN_COMMANDS: frozenset[str] = frozenset()


def _run_agent_macro(provider: str | None, model: str | None, **options: Any) -> None:
    handoff = plugins.load(OPERATOR_ENTRY_POINT_GROUP, "agent_macro")
    if not callable(handoff):
        raise click.ClickException(
            "Invalid agent macro plugin; reinstall the mobilerun-agents add-on."
        )
    handoff(provider=provider, model=model, **options)


def reserved_command_names() -> set[str]:
    """Names owned by core commands."""
    return set(_COMMANDS)


class LazyGroup(MenuGroup):
    def _plugin_commands(self, ctx: click.Context) -> dict:
        if "cli_plugin_commands" in ctx.meta:
            return ctx.meta["cli_plugin_commands"]
        entries = plugins.discover(CLI_ENTRY_POINT_GROUP)
        reserved = reserved_command_names()
        for name in sorted(entries.keys() & reserved):
            logging.getLogger(__name__).warning(
                "Skipping plugin command %s: built-in wins", name
            )
            del entries[name]
        ctx.meta["cli_plugin_commands"] = entries
        return entries

    def menu_choices(self, ctx: click.Context) -> list[SelectChoice]:
        return [
            SelectChoice(
                name,
                name,
                _COMMANDS.get(
                    name,
                    (
                        None,
                        None,
                        CLI_PLUGIN_HELP.get(name, "Installed plugin command."),
                    ),
                )[2],
            )
            for name in self.list_commands(ctx)
        ]

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(
            (set(_COMMANDS) - _HIDDEN_COMMANDS) | self._plugin_commands(ctx).keys()
        )

    def get_epilog(self, ctx: click.Context) -> str | None:
        epilog = super().get_epilog(ctx)
        if not epilog or "agent" in self._plugin_commands(ctx):
            return epilog
        return "; ".join(
            part
            for part in epilog.split("; ")
            if not part.startswith("mobilerun agent ")
        )

    def shell_complete(
        self, ctx: click.Context, incomplete: str
    ) -> list[CompletionItem]:
        """Complete registered names without importing their command modules."""
        names = [
            CompletionItem(choice.value, help=choice.hint)
            for choice in self.menu_choices(ctx)
            if choice.value.startswith(incomplete)
        ]
        return names + click.Command.shell_complete(self, ctx, incomplete)

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        rows = [
            (
                name,
                _COMMANDS.get(
                    name,
                    (
                        None,
                        None,
                        CLI_PLUGIN_HELP.get(name, "Installed plugin command."),
                    ),
                )[2],
            )
            for name in self.list_commands(ctx)
        ]
        with formatter.section("Commands"):
            formatter.write_dl(rows)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        target = _COMMANDS.get(cmd_name)
        if target is None:
            if cmd_name not in self._plugin_commands(ctx):
                if cmd_name == "agent":
                    return cast(
                        click.Command, plugins.load(CLI_ENTRY_POINT_GROUP, cmd_name)
                    )
                return None
            ctx.meta["macro_agent_handoff"] = _run_agent_macro
            command = plugins.load(CLI_ENTRY_POINT_GROUP, cmd_name)
            if not isinstance(command, click.Command):
                raise click.ClickException("Invalid CLI plugin; reinstall the add-on.")
            return command
        module, attribute, _short_help = target
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
