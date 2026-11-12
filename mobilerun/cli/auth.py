"""Sign-in commands using the existing credential flows."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import click
from rich.console import Console

from mobilerun.cli import deviceauth
from mobilerun.cli.options import (
    DEFAULT_AUTH_URL,
    Group,
    auth_url_option,
    emit,
    examples,
    input_allowed,
    json_option,
    no_input_option,
    read_secret,
    verbose_option,
)
from mobilerun.cli.qr import print_qr
from mobilerun.cli.selection import CLOUD_CREDENTIAL_LABELS, resolve_cloud_credential
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
    from mobilerun.cli.oauth_actions import run_openai_oauth_login

    run_openai_oauth_login(credential_path=credential_path, model=model, **kwargs)
    _print_oauth_login_success("OpenAI", credential_path)


def _run_gemini_oauth_login(credential_path: str, model: str | None, **kwargs) -> None:
    from mobilerun.cli.oauth_actions import run_gemini_oauth_login

    run_gemini_oauth_login(credential_path=credential_path, model=model, **kwargs)
    _print_oauth_login_success("Gemini", credential_path)


def _run_anthropic_oauth_login(credential_path: str, **kwargs) -> None:
    """Run the full Anthropic OAuth flow inline and save the token."""
    from mobilerun.cli.oauth_actions import run_anthropic_setup_token_oauth

    console.print("[blue]Starting Anthropic login...[/]")
    run_anthropic_setup_token_oauth(
        credential_path=credential_path,
        **kwargs,
    )
    _print_oauth_login_success("Anthropic", credential_path)


def _run_grok_oauth_login(credential_path: str, model: str | None = None, **kwargs) -> None:
    from mobilerun.cli.oauth_actions import run_grok_oauth_login

    run_grok_oauth_login(
        credential_path=credential_path,
        model=model,
        **kwargs,
    )
    _print_oauth_login_success("XAI", credential_path)


DEFAULT_DEVICE_CLIENT_ID = "mobilerun-cli"


def _login_mobilerun(auth_url: str | None, client_id: str, no_qr: bool, browser: bool):
    """Authenticate via OAuth 2.0 device authorization and save the session token."""
    import webbrowser

    try:
        code = deviceauth.request_code(auth_url, client_id)
    except Exception as e:
        raise click.ClickException(f"request device code: {e}\nCheck --auth-url or MOBILERUN_AUTH_URL: {auth_url}") from e

    link = code.verification_uri_complete
    terminal = sys.stderr.isatty()
    styled = terminal and "NO_COLOR" not in os.environ
    if link:
        if terminal and not no_qr:
            print_qr(link)
        display = f"\033]8;;{link}\033\\{link}\033]8;;\033\\" if styled else link
        click.echo(display, err=True, color=styled)
    click.echo(f"Visit: {code.verification_uri}", err=True)
    click.echo(f"Code:  {code.user_code}", err=True)
    if browser and link:
        try:
            webbrowser.open(link)
        except Exception:
            pass
    waiting = f"Waiting for authorization (expires in {code.expires_in}s)..."
    console.print(f"[dim]{waiting}[/]" if styled else waiting)

    try:
        token = deviceauth.poll_token(auth_url, client_id, code)
    except deviceauth.DeviceAuthError as e:
        raise click.ClickException(f"authorization failed: {e}") from e

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
LOGIN_OPTIONS = {
    "mobilerun": {"auth_url", "client_id", "no_qr", "browser", "no_browser"},
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

service_argument = click.argument(
    "service", type=click.Choice(["mobilerun", *SERVICES]), default="mobilerun", metavar="[SERVICE]"
)


@click.group(cls=Group, epilog=examples('mobilerun auth status'))
@verbose_option
def auth_cli(verbose):
    """Sign in, sign out, and inspect saved credentials."""


@auth_cli.command(
    epilog=examples(
        'mobilerun auth login',
        'mobilerun auth login --browser',
        'mobilerun auth login openai --no-browser',
    ),
)
@service_argument
@verbose_option
@auth_url_option
@click.option("-m", "--model", help="LLM model (openai, gemini, xai).")
@click.option("--client-id", default=None, help="Mobilerun device authorization client id.")
@click.option(
    "--credential-path", type=click.Path(dir_okay=False),
    help="Provider credential file (openai, anthropic, gemini, xai)."
)
@click.option(
    "--timeout",
    type=float,
    default=300.0,
    callback=_validate_oauth_timeout,
    help="Provider login timeout in seconds (openai, anthropic, gemini, xai).",
)
@click.option(
    "--no-browser", is_flag=True,
    help="Do not open a provider login browser (openai, anthropic, gemini, xai); "
         "unnecessary for mobilerun, which does not open one by default.",
)
@click.option("--browser", is_flag=True, help="Open the Mobilerun sign-in link in the browser.")
@click.option("--no-qr", is_flag=True, help="Do not display the Mobilerun sign-in QR code.")
@click.option("--callback-host", help="OAuth callback host (openai, anthropic, gemini).")
@click.option(
    "--callback-port", type=click.IntRange(0, 65535),
    help="OAuth callback port (openai, anthropic, gemini).",
)
@click.option("--callback-path", help="OAuth callback path (openai, anthropic, gemini).")
@click.option("--device-code", is_flag=True, help="Use xAI device-code login.")
@no_input_option
@click.option(
    "--token", callback=read_secret,
    help="Anthropic setup-token; prefer '-' for stdin, "
         "because flag values end up in shell history.",
)
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
    browser,
    no_qr,
    callback_host,
    callback_port,
    callback_path,
    device_code,
    token,
    no_input,
):
    """Log in to a service; SERVICE defaults to mobilerun."""
    from mobilerun.cli.oauth_actions import login_requires_input, save_anthropic_setup_token

    for name in set().union(*LOGIN_OPTIONS.values()):
        if (
            ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE
            and name not in LOGIN_OPTIONS[service]
        ):
            raise click.UsageError(f"--{name.replace('_', '-')} is not supported for {service}")
    if not input_allowed(no_input) and login_requires_input(service):
        if service == "anthropic" and not token:
            raise click.UsageError(
                "Missing option '--token'; pass --token - to read a setup-token from stdin."
            )
        if service == "gemini":
            raise click.UsageError(
                "Gemini login needs a code; omit --no-input and run "
                "`mobilerun auth login gemini` in a terminal."
            )
    if service == "mobilerun":
        _login_mobilerun(
            auth_url, client_id or DEFAULT_DEVICE_CLIENT_ID, no_qr, browser
        )
        credential = resolve_cloud_credential()
        if credential.source == "env":
            click.echo(
                f"Warning: {credential.env_var} is set and takes precedence over this sign-in; "
                "unset it to use the sign-in.", err=True,
            )
        return
    kwargs = {
        "credential_path": credential_path or str(SERVICES[service][0]),
        "timeout": timeout,
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


@auth_cli.command( epilog=examples('mobilerun auth logout'))
@service_argument
@verbose_option
def logout(service, verbose):
    """Remove SERVICE's stored credential, preserving other services."""
    from mobilerun.cli.configure_wizard import _OAUTH_CREDENTIAL_SLOTS, _oauth_credentials_present

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
    if service == "mobilerun":
        credential = resolve_cloud_credential()
        if credential.source == "env":
            console.print(f"{credential.env_var} stays active; unset it to stop using the env key.")


@auth_cli.command( epilog=examples('mobilerun auth status -j'))
@verbose_option
@json_option
def status(verbose, json_output):
    """Show saved sign-in status without exposing tokens."""
    from mobilerun.cli.configure_wizard import _oauth_credentials_present

    credentials = deviceauth.load_credentials() or {}
    token = credentials.get("access_token")
    credential = resolve_cloud_credential()
    rows = [
        {
            "service": "mobilerun",
            "signed_in": bool(token),
            "credential_path": str(deviceauth.credential_path()),
            "email": None,
            "cloud_credential": credential.source,
            "env_overrides_login": credential.source == "env" and credential.saved_sign_in,
        }
    ]
    failed = False
    if token:
        try:
            session = deviceauth.fetch_session(
                credentials.get("auth_url") or DEFAULT_AUTH_URL, token
            )
            if session is None:
                failed = True
            else:
                rows[0]["email"] = session.get("user", {}).get("email")
        except Exception:
            failed = True
    for service, (path, variant) in SERVICES.items():
        rows.append(
            {
                "service": service,
                "signed_in": _oauth_credentials_present(str(path), variant),
                "credential_path": str(path),
            }
        )
    if failed:
        raise click.ClickException(
            "Mobilerun session check failed; run `mobilerun auth login` again."
        )
    if json_output:
        emit(rows, json_output=True)
    else:
        cloud_note = (
            "; cloud commands have no credential; run `mobilerun auth login` "
            "or set MOBILERUN_API_KEY"
            if credential.source is None
            else f"; cloud commands use {CLOUD_CREDENTIAL_LABELS[credential.source]}"
        )
        if credential.source == "env" and credential.saved_sign_in:
            cloud_note += " (overrides saved sign-in)"
        for row in rows:
            label = "signed in" if row["signed_in"] else "not signed in"
            email = f" ({row['email']})" if row.get("email") else ""
            cloud = cloud_note if row["service"] == "mobilerun" else ""
            emit(f"{row['service']}: {label}{email}  {row['credential_path']}{cloud}")
