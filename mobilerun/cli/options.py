"""Shared Click options and command IO policy."""

import asyncio
import json
import sys
from contextlib import redirect_stdout
from functools import wraps

import click

DEFAULT_API_URL = "https://api.mobilerun.ai/v1"
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


class Command(click.Command):
    def invoke(self, ctx):
        ctx.ensure_object(dict)
        ctx.obj["stdout"] = sys.stdout
        if "verbose" in ctx.params and ctx.params["verbose"] is None:
            parent = ctx.parent
            while parent:
                if parent.params.get("verbose") is not None:
                    ctx.params["verbose"] = parent.params["verbose"]
                    break
                parent = parent.parent
        import logging

        from mobilerun.log_handlers import configure_logging

        logger = logging.getLogger("mobilerun")
        previous = logger.handlers[:], logger.level, logger.propagate
        configure_logging(bool(ctx.params.get("verbose")), logging.StreamHandler(sys.stderr))
        try:
            # Libraries and OAuth flows may print progress directly.
            with redirect_stdout(sys.stderr):
                return super().invoke(ctx)
        except (KeyboardInterrupt, asyncio.CancelledError, click.Abort):
            ctx.exit(130)
        except (click.ClickException, SystemExit):
            raise
        except Exception as exc:
            raise click.ClickException(str(exc) or type(exc).__name__) from exc
        finally:
            logger.handlers, level, logger.propagate = previous
            logger.setLevel(level)


class Group(click.Group):
    command_class = Command
    group_class = type

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("context_settings", CONTEXT_SETTINGS)
        super().__init__(*args, **kwargs)


# The same declarations are reused everywhere, including groups.
verbose_option = click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=None,
    envvar="MOBILERUN_DEBUG",
    show_envvar=True,
    help="Enable verbose logs.",
)
config_option = click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(dir_okay=False),
    envvar="MOBILERUN_CONFIG",
    show_envvar=True,
    help="Config file.",
)
device_option = click.option(
    "-d",
    "--device",
    envvar="MOBILERUN_DEVICE",
    show_envvar=True,
    help="ADB serial, Portal URL, or cloud device id.",
)
target_option = click.option(
    "-t",
    "--target",
    type=click.Choice(["local", "cloud"]),
    default="local",
    envvar="MOBILERUN_TARGET",
    show_envvar=True,
    show_default=True,
    help="Where to look for devices when no device is specified.",
)
platform_option = click.option(
    "-P",
    "--platform",
    type=click.Choice(["android", "ios"]),
    default=None,
    help="Device platform (otherwise use config).",
)
api_url_option = click.option(
    "--api-url",
    default=DEFAULT_API_URL,
    envvar="MOBILERUN_API_BASE_URL",
    show_envvar=True,
    show_default=True,
    help="Cloud API URL.",
)
auth_url_option = click.option(
    "--auth-url",
    default=None,
    envvar="MOBILERUN_AUTH_URL",
    show_envvar=True,
    help="Sign-in URL (default https://cloud.mobilerun.ai/api/auth).",
)
tcp_option = click.option("--tcp/--no-tcp", default=None, help="Use local Android TCP.")
json_option = click.option(
    "-j", "--json", "json_output", is_flag=True, help="Print one JSON document."
)
model_option = click.option("-m", "--model", help="LLM model.")
llm_url_option = click.option("-u", "--llm-url", help="LLM API URL.")
max_steps_option = click.option("-s", "--max-steps", type=int, help="Maximum steps.")


def device_options(function):
    for option in (
        device_option,
        target_option,
        platform_option,
        config_option,
        api_url_option,
        tcp_option,
        verbose_option,
    ):
        function = option(function)
    return function


def coro(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


def emit(value, *, json_output=False):
    """Write data to the original stdout, even inside a progress redirect."""
    ctx = click.get_current_context(silent=True)
    output = ctx.obj.get("stdout") if ctx and ctx.obj else sys.stdout
    click.echo(json.dumps(value, default=str) if json_output else value, file=output)
