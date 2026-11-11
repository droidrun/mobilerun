"""Shared Click options and command IO policy."""

import asyncio
import difflib
import json
import os
import sys
import traceback
from contextlib import redirect_stdout
from functools import wraps

import click
from mobilerun_sdk import APIConnectionError, APIStatusError, APITimeoutError

DEFAULT_API_URL = "https://api.mobilerun.ai/v1"
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _sdk_error(exc):
    while exc is not None:
        if isinstance(exc, (APIStatusError, APIConnectionError)):
            return exc
        exc = exc.__cause__
    return None


def convert_api_error(exc):
    """Return safe text, JSON and method/path diagnostics for an SDK error."""
    exc = _sdk_error(exc)
    if exc is None:
        return None
    request = exc.request
    sensitive = [os.environ.get("MOBILERUN_CLOUD_API_KEY", "")]
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        sensitive.append(authorization[7:])

    def clean(value):
        text = " ".join(str(value).split())
        for secret in sorted(filter(None, sensitive), key=len, reverse=True):
            text = text.replace(secret, "[redacted]")
        return text

    status = exc.status_code if isinstance(exc, APIStatusError) else None
    body = exc.body if isinstance(exc.body, dict) else {}
    first = (body.get("errors") or [None])[0] if isinstance(body.get("errors"), list) else None
    fields = (body, first) if isinstance(first, dict) else (body,)
    message = next((entry[key] for entry in fields for key in ("detail", "title", "message")
                    if isinstance(entry.get(key), str) and entry[key].strip()), None)
    code = next((entry["code"] for entry in fields
                 if isinstance(entry.get("code"), (str, int))), None)
    if message is None:
        if status is not None:
            message = exc.response.reason_phrase
        elif isinstance(exc, APITimeoutError):
            message = "Request timed out"
        else:
            message = "Could not connect to the API"
    message = clean(message or "API request failed")
    document = {"error": message, "status": status, "code": clean(code) if code is not None else None}
    text = f"{message} (HTTP {status})" if status is not None else message
    hint = None
    if status in {401, 403}:
        hint = "Run `mobilerun auth login` or set MOBILERUN_CLOUD_API_KEY."
    elif status == 404 and "devices" in request.url.path.split("/"):
        hint = "Find device ids with `mobilerun device list -t cloud`."
    elif status is None:
        hint = f"Check the API URL: {clean(request.url.scheme + '://' + request.url.host)}"
    if hint:
        text += f"\n{hint}"
    diagnostic = f"{request.method} {clean(request.url.path)}"
    return text, document, diagnostic


class HelpLayout:
    """Keep shared flags and complete command summaries consistent."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("context_settings", CONTEXT_SETTINGS)
        super().__init__(*args, **kwargs)
        for param in self.params:
            if isinstance(param, click.Option) and param.show_default is None:
                param.show_default = True

    def get_params(self, ctx):
        first = {"device": 0, "target": 1, "platform": 2, "config_path": 3}
        last = {"api_url": 5, "json_output": 6, "verbose": 7, "help": 8}
        return sorted(super().get_params(ctx), key=lambda param: first.get(
            param.name, last.get(param.name, 4)
        ))

    def get_short_help_str(self, limit=45):
        return self.short_help or " ".join((self.help or "").split("\n\n", 1)[0].split())


class Command(HelpLayout, click.Command):
    def invoke(self, ctx):
        ctx.ensure_object(dict)
        ctx.obj["stdout"] = sys.stdout
        ctx.obj["emitted"] = False
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
                from mobilerun.config_manager import ConfigLoader

                if ctx.command.name != "doctor":
                    ConfigLoader.resolve_path(ctx.params.get("config_path"))
                return super().invoke(ctx)
        except (KeyboardInterrupt, asyncio.CancelledError, click.Abort):
            ctx.exit(130)
        except Exception as exc:
            converted = convert_api_error(exc)
            if converted:
                text, document, diagnostic = converted
                if ctx.params.get("verbose"):
                    click.echo(diagnostic, err=True)
                exc = click.ClickException(text)
            else:
                if not isinstance(exc, click.ClickException):
                    if ctx.params.get("verbose"):
                        traceback.print_exc(file=sys.stderr)
                    exc = click.ClickException(str(exc) or type(exc).__name__)
                document = {"error": exc.message}
            if (not isinstance(exc, click.UsageError) and ctx.params.get("json_output")
                    and not ctx.obj["emitted"]):
                emit(document, json_output=True)
            raise exc
        finally:
            logger.handlers, level, logger.propagate = previous
            logger.setLevel(level)


class Group(HelpLayout, click.Group):
    command_class = Command
    group_class = type

    def __init__(self, *args, command_order=None, **kwargs):
        self.command_order = command_order
        super().__init__(*args, **kwargs)

    def list_commands(self, ctx):
        names = super().list_commands(ctx)
        if self.command_order:
            return [name for name in self.command_order if name in names] + [
                name for name in names if name not in self.command_order
            ]
        return names

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as exc:
            if args and exc.message.startswith("No such command"):
                matches = difflib.get_close_matches(args[0], self.list_commands(ctx), n=1, cutoff=0.6)
                if matches:
                    exc.message += f" Did you mean '{matches[0]}'?"
            raise


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
    default="https://cloud.mobilerun.ai/api/auth",
    envvar="MOBILERUN_AUTH_URL",
    show_envvar=True,
    help="Mobilerun sign-in URL.",
)
tcp_option = click.option("--tcp/--no-tcp", default=None, help="Use local Android TCP.")
json_option = click.option(
    "-j", "--json", "json_output", is_flag=True, help="Print one JSON document."
)
def model_option(function=None, *, help="LLM model."):
    decorator = click.option("-m", "--model", help=help)
    return decorator(function) if function is not None else decorator


llm_url_option = click.option("-u", "--llm-url", help="LLM API URL.")
max_steps_option = click.option("-s", "--max-steps", type=int, help="Maximum steps.")


def device_options(function):
    for option in (
        device_option,
        target_option,
        platform_option,
        config_option,
        api_url_option,
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
    if ctx and ctx.obj is not None and json_output:
        ctx.obj["emitted"] = True
