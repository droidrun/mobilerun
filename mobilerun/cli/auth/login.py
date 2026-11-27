"""Cloud device login command."""

import logging
import webbrowser
from datetime import datetime, timezone

import click
from click.core import ParameterSource
from rich.console import Console

from mobilerun.cli.auth.status import mask_key
from mobilerun.cli.base import render
from mobilerun.cli.base.billing import billing_option
from mobilerun.cli.base.run import coro
from mobilerun.clients.cloud import api_keys as cloud
from mobilerun.clients.cloud import device_flow as deviceauth
from mobilerun.clients.local import login_store as credentials
from mobilerun.config.settings import get_settings
from mobilerun.errors import CloudAPIError, InvalidAPIKeyError
from mobilerun.errors.base import MobilerunError
from mobilerun.models.billing import Billing

console = Console()
logger = logging.getLogger(__name__)

DEFAULT_DEVICE_CLIENT_ID = "mobilerun-cli"


@click.command(short_help="Log in to Mobilerun Cloud")
@click.option(
    "--api-key", help="Visible in shell history; prefer MOBILERUN_API_KEY in CI"
)
@click.option(
    "--auth-url",
    "auth_url",
    default=None,
    help="Better-auth base URL (overrides MOBILERUN_AUTH_URL)",
)
@click.option(
    "--client-id",
    "client_id",
    default=DEFAULT_DEVICE_CLIENT_ID,
    show_default=True,
    help="OAuth device client id",
)
@billing_option
@coro
async def login(
    auth_url: str | None, client_id: str, api_key: str | None, billing: str
) -> None:
    """Authenticate via OAuth 2.0 device authorization and save the session token."""
    if api_key is not None:
        ctx = click.get_current_context()
        if any(
            ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE
            for name in ("client_id", "auth_url")
        ):
            raise click.UsageError(
                "--api-key cannot be combined with --client-id or --auth-url"
            )
        try:
            await cloud.validate_api_key(api_key, billing=Billing(billing))
        except (CloudAPIError, InvalidAPIKeyError) as exc:
            raise click.ClickException(str(exc)) from exc
        path = credentials.save_credentials(
            {
                "api_key": api_key,
                "auth_type": "api_key",
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        render.success(f"Logged in with API key {mask_key(api_key)}.")
        render.kv([("Credentials saved to", render.secondary(path))])
        return

    auth_url = auth_url or get_settings().MOBILERUN_AUTH_URL

    try:
        code = await deviceauth.request_code(
            auth_url, client_id, billing=Billing(billing)
        )
    except MobilerunError as e:
        raise click.ClickException(f"request device code: {e}") from e

    console.print(f"Visit: [bold]{code.verification_uri}[/]")
    console.print(f"Code:  [bold cyan]{code.user_code}[/]")
    if code.verification_uri_complete:
        console.print(f"[dim](Opening {code.verification_uri_complete} ...)[/]")
        try:
            webbrowser.open(code.verification_uri_complete)
        except (OSError, webbrowser.Error):
            logger.warning(
                "Could not open the browser; use the verification URL above."
            )
    console.print(
        f"[dim]Waiting for authorization (expires in {code.expires_in}s)...[/]"
    )

    try:
        token = await deviceauth.poll_token(
            auth_url, client_id, code, billing=Billing(billing)
        )
    except (deviceauth.DeviceAuthError, CloudAPIError) as e:
        raise click.ClickException(f"authorization failed: {e}") from e
    except KeyboardInterrupt:
        raise click.ClickException("login cancelled") from None

    path = credentials.save_token(token, auth_url)
    email = None
    try:
        info = await deviceauth.fetch_session(auth_url, token, billing=Billing(billing))
        if info:
            email = info.get("user", {}).get("email")
    except (deviceauth.DeviceAuthError, CloudAPIError, ValueError):
        logger.warning("Could not retrieve the account name for this session.")
    who = f" as {email}" if email else ""
    render.success(f"Logged in{who}.")
    render.kv([("Token saved to", render.secondary(path))])
