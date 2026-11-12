"""Shared Click options and command IO policy."""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import sys
import traceback
import unicodedata
from contextlib import redirect_stdout
from functools import wraps

import click

DEFAULT_API_URL = "https://api.mobilerun.ai/v1"
DEFAULT_AUTH_URL = "https://cloud.mobilerun.ai/api/auth"
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "show_default": True}
SECRETS_KEY = "cli_secrets"
NO_LOCAL_DEVICES_HINT = (
    "Connect a device over USB or run `mobilerun device pair`; "
    "use `-t cloud` for cloud devices."
)

NO_CLOUD_DEVICES_HINT = "Create or start a device in Mobilerun Cloud and check your sign-in."


ADB_HINT = (
    "adb was not found or is not running. Install Android SDK Platform-Tools "
    "(https://developer.android.com/studio/releases/platform-tools) "
    "and check `mobilerun doctor`."
)


def config_debug(path):
    """Read logging policy without creating or migrating configuration."""
    import yaml

    try:
        with path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        return data.get("logging", {}).get("debug") is True if isinstance(data, dict) else False
    except (OSError, ValueError, AttributeError, yaml.YAMLError):
        return False


def local_device_hint(message, target="local"):
    """Add the recovery step for known local discovery errors."""
    lowered = message.lower()
    if any(text in lowered for text in (
        "no connected android devices", "no devices", "no device connected",
    )):
        return message + "\n" + (NO_CLOUD_DEVICES_HINT if target == "cloud" else NO_LOCAL_DEVICES_HINT)
    if "unknown device" in lowered or ("device" in lowered and "not found" in lowered):
        command = "mobilerun device list -t cloud" if target == "cloud" else "mobilerun device list"
        return message + f"\nFind connected devices with `{command}`."
    return message


def read_secret(ctx, param, value):
    """Accept one secret line from stdin, without echoing it."""
    if ctx.resilient_parsing:
        return value
    if value == "-":
        stream = click.get_text_stream("stdin")
        try:
            value = (click.prompt(param.human_readable_name, hide_input=True, err=True,
                                  default="", show_default=False)
                     if stream.isatty() else stream.readline()).rstrip("\r\n")
        except (KeyboardInterrupt, click.Abort) as exc:
            if isinstance(exc, KeyboardInterrupt) or isinstance(
                exc.__cause__ or exc.__context__, KeyboardInterrupt
            ):
                raise click.exceptions.Exit(130) from None
            raise click.ClickException("Input closed; run the command again in a terminal.") from None
        if not value.strip():
            raise click.BadParameter("stdin is empty; provide one secret line.", ctx=ctx, param=param)
    if value:
        ctx.meta.setdefault(SECRETS_KEY, []).append(value)
    return value


def redact(text, secrets):
    for secret in sorted({value for value in secrets if value and len(value) >= 8}, key=len, reverse=True):
        text = text.replace(secret, "[redacted]")
    return text


def redact_secrets(text, ctx):
    return redact(text, ctx.meta.get(SECRETS_KEY, ()))


class SecretFormatter(logging.Formatter):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx

    def format(self, record):
        return redact_secrets(super().format(record), self.ctx)


def input_allowed(no_input=False):
    return not no_input and sys.stdin.isatty()


def _sdk_error(exc):
    from mobilerun_sdk import APIConnectionError, APIStatusError

    while exc is not None:
        if isinstance(exc, (APIStatusError, APIConnectionError)):
            return exc
        exc = exc.__cause__
    return None


def convert_api_error(exc):
    """Return safe text, JSON and method/path diagnostics for an SDK error."""
    from mobilerun_sdk import APIStatusError, APITimeoutError

    exc = _sdk_error(exc)
    if exc is None:
        return None
    request = exc.request
    from mobilerun_core.connection.cloud import BEARER_ENV_VAR, LEGACY_BEARER_ENV_VAR

    sensitive = [os.environ.get(name, "") for name in (BEARER_ENV_VAR, LEGACY_BEARER_ENV_VAR)]
    ctx = click.get_current_context(silent=True)
    if ctx:
        sensitive.extend(ctx.meta.get(SECRETS_KEY, ()))
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        sensitive.append(authorization[7:])

    def clean(value):
        return redact(" ".join(str(value).split()), sensitive)

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
        from mobilerun.cli.selection import resolve_cloud_credential

        credential = resolve_cloud_credential()
        document["credential"] = credential.source or "login"
        if credential.source == "env":
            hint = f"The key in {credential.env_var} was rejected. "
            if credential.saved_sign_in:
                hint += "A saved sign-in exists; unset the variable to use it."
            else:
                hint += "Check the key, or unset the variable and run `mobilerun auth login`."
        else:
            hint = "The saved sign-in was rejected or has expired. Run `mobilerun auth login`."
    elif status == 404 and "devices" in request.url.path.split("/"):
        hint = "Find device ids with `mobilerun device list -t cloud`."
    elif status is None:
        host = request.url.host
        if ":" in host:
            host = f"[{host}]"
        origin = f"{request.url.scheme}://{host}"
        if request.url.port is not None:
            origin += f":{request.url.port}"
        hint = f"Check the API URL from --api-url or MOBILERUN_API_BASE_URL: {clean(origin)}."
    if hint:
        text += f"\n{hint}"
    diagnostic = f"{request.method} {clean(request.url.path)}"
    return text, document, diagnostic


class HelpLayout:
    """Keep shared flags and complete command summaries consistent."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("context_settings", CONTEXT_SETTINGS)
        super().__init__(*args, **kwargs)

    def parse_args(self, ctx, args):
        if not ctx.resilient_parsing:
            for arg in args:
                if arg == "--":
                    break
                flag = arg.split("=", 1)[0]
                replacement = _REMOVED_FLAGS.get(flag)
                if flag == "--path" and self.name == "setup":
                    replacement = "--apk"
                if replacement:
                    raise click.UsageError(
                        f"Removed option: {flag}. Use {replacement} instead.", ctx
                    )
        try:
            return super().parse_args(ctx, args)
        except click.NoSuchOption as exc:
            if self.name in {"tap", "type", "swipe", "long-press"}:
                exc.message += " Put `--` before arguments that start with a dash."
            raise

    def get_params(self, ctx):
        def rank(param):
            order = ("device", "target", "platform", "config_path", None,
                     "api_url", "json_output", "verbose", "help")
            return order.index(param.name if param.name in order else None)

        return sorted(super().get_params(ctx), key=rank)

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
        from mobilerun.log_handlers import configure_logging

        logger = logging.getLogger("mobilerun")
        previous = logger.handlers[:], logger.level, logger.propagate
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(SecretFormatter(ctx))
        try:
            # Libraries and OAuth flows may print progress directly.
            with redirect_stdout(sys.stderr):
                if ctx.command.name != "completion":
                    from mobilerun.config_manager import ConfigLoader

                    try:
                        path = ConfigLoader.resolve_path(ctx.params.get("config_path"))
                    except (OSError, ValueError):
                        if ctx.command.name != "doctor":
                            raise
                        path = ConfigLoader.get_user_config_path()
                else:
                    from pathlib import Path

                    from mobilerun.config_manager.credential_paths import USER_CONFIG_DIR

                    path = Path(os.environ.get("MOBILERUN_CONFIG", USER_CONFIG_DIR / "config.yaml"))
                if ctx.params.get("verbose") is None:
                    ctx.params["verbose"] = config_debug(path)
                configure_logging(bool(ctx.params.get("verbose")), handler)
                return super().invoke(ctx)
        except (KeyboardInterrupt, asyncio.CancelledError):
            ctx.exit(130)
        except (EOFError, click.Abort) as exc:
            if isinstance(exc.__cause__ or exc.__context__, KeyboardInterrupt):
                ctx.exit(130)
            raise click.ClickException("Input closed; run the command again in a terminal.") from None
        except click.exceptions.Exit:
            raise
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
                        click.echo(redact_secrets(traceback.format_exc(), ctx), err=True)
                    exc = click.ClickException(str(exc) or type(exc).__name__)
                exc.message = local_device_hint(
                    redact_secrets(exc.message, ctx), ctx.params.get("target", "local"),
                )
                document = {"error": exc.message}
            if (not isinstance(exc, click.UsageError) and ctx.params.get("json_output")
                    and not ctx.obj["emitted"]):
                emit(document, json_output=True)
            raise exc
        finally:
            logger.handlers, level, logger.propagate = previous
            logger.setLevel(level)


_REMOVED_FLAGS = {
    "--base_url": "-u/--llm-url", "--api_base": "-u/--llm-url",
    "--cloud": "-t cloud", "--ios": "-P ios", "--debug": "-v",
    "--device-id": "-d", "--steps": "-s/--max-steps",
    "-pv": "--portal-version", "--control-backend": "--backend",
}

_RENAMED_COMMANDS = {
    "devices": "mobilerun device list",
    "whoami": "mobilerun auth status",
}


def _command_paths(command, path=("mobilerun",)):
    """Enumerate registered command paths without invoking callbacks."""
    yield path, command
    if isinstance(command, click.Group):
        for name, child in command.commands.items():
            if not child.hidden:
                yield from _command_paths(child, (*path, name))


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
            if not ctx.resilient_parsing and args and exc.message.startswith("No such command"):
                matches = self.suggest_commands(ctx, args[0])
                if matches:
                    paths = {" ".join(path): command for path, command in
                             _command_paths(ctx.find_root().command)}
                    suggestions = []
                    for match in matches:
                        command = paths.get(match)
                        for remaining in args[1:]:
                            if not isinstance(command, click.Group) or remaining not in command.commands:
                                break
                            match += " " + remaining
                            command = command.commands[remaining]
                        suggestions.append(f"'{match}'")
                    suggestions = ", ".join(suggestions)
                    exc.message += f" Did you mean {suggestions}?"
            raise

    def suggest_commands(self, ctx, name):
        paths = list(_command_paths(ctx.find_root().command))
        exact = [" ".join(path) for path, _ in paths if path[-1] == name]
        if exact:
            return exact
        if name in _RENAMED_COMMANDS:
            return [_RENAMED_COMMANDS[name]]
        siblings = difflib.get_close_matches(name, self.list_commands(ctx), n=1, cutoff=0.6)
        if siblings:
            parent_path = next(path for path, command in paths if command is self)
            return [" ".join((*parent_path, siblings[0]))]
        closest = difflib.get_close_matches(name, [path[-1] for path, _ in paths], n=1, cutoff=0.6)
        if closest:
            return [next(" ".join(path) for path, _ in paths if path[-1] == closest[0])]
        return []


# The same declarations are reused everywhere, including groups.
no_input_option = click.option("--no-input", is_flag=True, help="Never prompt for input.")
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
    default=DEFAULT_AUTH_URL,
    envvar="MOBILERUN_AUTH_URL",
    show_envvar=True,
    help="Mobilerun sign-in URL.",
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
    output = stdout()
    click.echo(json.dumps(value, default=str) if json_output else value, file=output)
    if ctx and ctx.obj is not None and json_output:
        ctx.obj["emitted"] = True


def _ui_cell(value):
    return "".join(
        " " if unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"} else char
        for char in str(value if value is not None else "")
    )


def emit_rows(headers, rows):
    """Print tab-separated data, adding a header only on terminal stdout."""
    output = stdout()

    def line(values):
        return "\t".join(_ui_cell(value) for value in values)

    if output.isatty():
        emit(line(headers))
    for row in rows:
        emit(line(row))


def stdout():
    ctx = click.get_current_context(silent=True)
    return ctx.obj.get("stdout", sys.stdout) if ctx and ctx.obj else sys.stdout


def examples(*lines, docs=None):
    text = "\b\nExamples:\n" + "\n".join("  " + line for line in lines)
    return text + (f"\n\nDocs: {docs}" if docs else "")



def _lazy_imports(namespace, exports):
    """Preserve module export paths while deferring optional dependencies."""
    from importlib import import_module

    def get_attribute(name):
        if name not in exports:
            raise AttributeError(f"module {namespace['__name__']!r} has no attribute {name!r}")
        module, attribute = exports[name]
        value = import_module(module)
        if attribute is not None:
            value = getattr(value, attribute)
        namespace[name] = value
        return value

    return get_attribute

__getattr__ = _lazy_imports(globals(), {
    'APIConnectionError': ('mobilerun_sdk', 'APIConnectionError'),
    'APIStatusError': ('mobilerun_sdk', 'APIStatusError'),
    'APITimeoutError': ('mobilerun_sdk', 'APITimeoutError'),
})

from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from mobilerun_sdk import APIConnectionError as APIConnectionError
    from mobilerun_sdk import APIStatusError as APIStatusError
    from mobilerun_sdk import APITimeoutError as APITimeoutError
