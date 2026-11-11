"""
Mobilerun CLI - Command line interface for controlling Android devices through LLM agents.
"""

import math
import os
from pathlib import Path

import click
from rich.console import Console

from mobilerun.cli import deviceauth
from mobilerun.cli.configure_wizard import _OAUTH_CREDENTIAL_SLOTS, _oauth_credentials_present
from mobilerun.cli.oauth_actions import (
    run_anthropic_setup_token_oauth,
    run_gemini_oauth_login,
    run_grok_oauth_login,
    run_openai_oauth_login,
    save_anthropic_setup_token,
)
from mobilerun.cli.options import (
    Group,
    auth_url_option,
    emit,
    json_option,
    model_option,
    verbose_option,
)
from mobilerun.config_manager.auth_profile_store import AuthProfileStore
from mobilerun.config_manager.credential_paths import (
    ANTHROPIC_OAUTH_CREDENTIAL_PATH,
    GEMINI_OAUTH_CREDENTIAL_PATH,
    GROK_OAUTH_CREDENTIAL_PATH,
    OPENAI_OAUTH_CREDENTIAL_PATH,
)

console = Console(stderr=True)


def _validate_oauth_timeout(
    _ctx: click.Context,
    _param: click.Parameter,
    value: float | None,
) -> float:
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise click.BadParameter("must be a finite number greater than zero")
    return value


def _print_oauth_login_success(provider_label: str, credential_path: str) -> None:
    console.print(f"[green]{provider_label} login succeeded.[/]")
    console.print(f"[blue]Credentials saved to:[/] {Path(credential_path).expanduser()}")


def _run_openai_oauth_login(credential_path: str, model: str | None, **kwargs) -> None:
    run_openai_oauth_login(credential_path=credential_path, model=model, **kwargs)
    _print_oauth_login_success("OpenAI", credential_path)


def _run_gemini_oauth_login(credential_path: str, model: str | None, **kwargs) -> None:
    run_gemini_oauth_login(credential_path=credential_path, model=model, **kwargs)
    _print_oauth_login_success("Gemini", credential_path)


def _run_anthropic_oauth_login(credential_path: str, **kwargs) -> None:
    """Run the full Anthropic OAuth flow inline and save the token."""
    console.print("[blue]Starting Anthropic login...[/]")
    run_anthropic_setup_token_oauth(
        credential_path=credential_path,
        **kwargs,
    )
    _print_oauth_login_success("Anthropic", credential_path)


def _run_grok_oauth_login(credential_path: str, model: str | None = None, **kwargs) -> None:
    run_grok_oauth_login(
        credential_path=credential_path,
        model=model,
        **kwargs,
    )
    _print_oauth_login_success("XAI", credential_path)


DEFAULT_AUTH_URL = "https://cloud.mobilerun.ai/api/auth"
DEFAULT_DEVICE_CLIENT_ID = "mobilerun-cli"


def _login_mobilerun(auth_url: str | None, client_id: str):
    """Authenticate via OAuth 2.0 device authorization and save the session token."""
    import webbrowser

    from mobilerun.cli import deviceauth

    auth_url = auth_url or os.getenv("MOBILERUN_AUTH_URL") or DEFAULT_AUTH_URL

    try:
        code = deviceauth.request_code(auth_url, client_id)
    except Exception as e:
        raise click.ClickException(f"request device code: {e}") from e

    console.print(f"Visit: [bold]{code.verification_uri}[/]")
    console.print(f"Code:  [bold cyan]{code.user_code}[/]")
    if code.verification_uri_complete:
        console.print(f"[dim](Opening {code.verification_uri_complete} ...)[/]")
        try:
            webbrowser.open(code.verification_uri_complete)
        except Exception:
            pass
    console.print(f"[dim]Waiting for authorization (expires in {code.expires_in}s)...[/]")

    try:
        token = deviceauth.poll_token(auth_url, client_id, code)
    except deviceauth.DeviceAuthError as e:
        raise click.ClickException(f"authorization failed: {e}") from e
    except KeyboardInterrupt:
        raise

    path = deviceauth.save_token(token, auth_url)
    email = None
    try:
        info = deviceauth.fetch_session(auth_url, token)
        if info:
            email = info.get("user", {}).get("email")
    except Exception:
        pass
    who = f" as {email}" if email else ""
    console.print(f"[green]Logged in{who}.[/] Token saved to {path}")


SERVICES = {
    "openai": (OPENAI_OAUTH_CREDENTIAL_PATH, "openai_oauth"),
    "anthropic": (ANTHROPIC_OAUTH_CREDENTIAL_PATH, "anthropic_oauth"),
    "gemini": (GEMINI_OAUTH_CREDENTIAL_PATH, "gemini_oauth_code_assist"),
    "xai": (GROK_OAUTH_CREDENTIAL_PATH, "xai_oauth"),
}
service_argument = click.argument(
    "service", type=click.Choice(["mobilerun", *SERVICES]), default="mobilerun"
)


@click.group(cls=Group)
@verbose_option
def auth_cli(verbose):
    """Sign in, sign out, and inspect saved credentials."""


@auth_cli.command()
@service_argument
@verbose_option
@auth_url_option
@model_option
@click.option("--client-id", default=None, help="Mobilerun device authorization client id.")
@click.option(
    "--credential-path", type=click.Path(dir_okay=False), help="Provider credential file."
)
@click.option(
    "--timeout",
    type=float,
    default=None,
    callback=_validate_oauth_timeout,
    help="Login timeout in seconds (default 300).",
)
@click.option("--no-browser", is_flag=True, help="Do not open a browser.")
@click.option("--callback-host", help="OAuth callback host.")
@click.option("--callback-port", type=click.IntRange(0, 65535), help="OAuth callback port.")
@click.option("--callback-path", help="OAuth callback path.")
@click.option("--device-code", is_flag=True, help="Use xAI device-code login.")
@click.option("--token", help="Save an Anthropic setup-token.")
@click.pass_context
def login(
    ctx,
    service,
    verbose,
    auth_url,
    model,
    client_id,
    credential_path,
    timeout,
    no_browser,
    callback_host,
    callback_port,
    callback_path,
    device_code,
    token,
):
    """Log in to SERVICE (default mobilerun)."""
    supported = {
        "mobilerun": {"auth_url", "client_id"},
        "openai": {
            "credential_path",
            "timeout",
            "no_browser",
            "callback_host",
            "callback_port",
            "callback_path",
            "model",
        },
        "anthropic": {
            "credential_path",
            "timeout",
            "no_browser",
            "callback_host",
            "callback_port",
            "callback_path",
            "token",
        },
        "gemini": {
            "credential_path",
            "timeout",
            "no_browser",
            "callback_host",
            "callback_port",
            "callback_path",
            "model",
        },
        "xai": {"credential_path", "timeout", "no_browser", "device_code", "model"},
    }
    for name in (
        "auth_url",
        "client_id",
        "credential_path",
        "timeout",
        "no_browser",
        "callback_host",
        "callback_port",
        "callback_path",
        "device_code",
        "token",
        "model",
    ):
        if (
            ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE
            and name not in supported[service]
        ):
            raise click.UsageError(f"--{name.replace('_', '-')} is not supported for {service}")
    if service == "mobilerun":
        _login_mobilerun(auth_url or DEFAULT_AUTH_URL, client_id or DEFAULT_DEVICE_CLIENT_ID)
        return
    kwargs = {
        "credential_path": credential_path or str(SERVICES[service][0]),
        "timeout": timeout or 300.0,
        "open_browser": not no_browser,
    }
    if service in {"openai", "gemini", "anthropic"}:
        for name, value in (
            ("callback_host", callback_host),
            ("callback_port", callback_port),
            ("callback_path", callback_path),
        ):
            if value is not None:
                kwargs[name] = value
    if service == "anthropic":
        if token:
            save_anthropic_setup_token(kwargs["credential_path"], token)
            _print_oauth_login_success("Anthropic", kwargs["credential_path"])
        else:
            _run_anthropic_oauth_login(**kwargs)
    else:
        kwargs["model"] = model
        if service == "xai":
            kwargs["device_code"] = device_code
        {
            "openai": _run_openai_oauth_login,
            "gemini": _run_gemini_oauth_login,
            "xai": _run_grok_oauth_login,
        }[service](**kwargs)


@auth_cli.command()
@service_argument
@verbose_option
def logout(service, verbose):
    """Remove SERVICE's stored credential, preserving other services."""
    if service == "mobilerun":
        removed = deviceauth.clear_credentials()
    else:
        path, variant = SERVICES[service]
        removed = _oauth_credentials_present(str(path), variant)
        if path.exists():
            AuthProfileStore(path).update_profile(
                lambda profile: profile.pop(_OAUTH_CREDENTIAL_SLOTS[variant], None)
            )
    console.print(f"Logged out of {service}." if removed else f"Not logged in to {service}.")


@auth_cli.command()
@verbose_option
@json_option
@auth_url_option
def status(verbose, json_output, auth_url):
    """Show saved sign-in status without exposing tokens."""
    credentials = deviceauth.load_credentials() or {}
    token = credentials.get("access_token")
    rows = [
        {
            "service": "mobilerun",
            "signed_in": bool(token),
            "credential_path": str(deviceauth.credential_path()),
            "email": None,
        }
    ]
    failed = False
    if token:
        try:
            session = deviceauth.fetch_session(
                auth_url or credentials.get("auth_url") or DEFAULT_AUTH_URL, token
            )
            if session is None:
                failed = True
            else:
                rows[0]["email"] = session.get("user", {}).get("email")
        except Exception:
            failed = True
        if failed:
            rows[0]["signed_in"] = False
            rows[0]["error"] = "session check failed"
    for service, (path, variant) in SERVICES.items():
        rows.append(
            {
                "service": service,
                "signed_in": _oauth_credentials_present(str(path), variant),
                "credential_path": str(path),
            }
        )
    if json_output:
        emit(rows, json_output=True)
    else:
        for row in rows:
            label = "signed in" if row["signed_in"] else "not signed in"
            email = f" ({row['email']})" if row.get("email") else ""
            emit(f"{row['service']}: {label}{email}  {row['credential_path']}")
    if failed:
        raise click.ClickException(
            "Mobilerun session check failed; run `mobilerun auth login` again."
        )
