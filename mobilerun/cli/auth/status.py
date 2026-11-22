"""Cloud login status command."""

import json

import click
from rich.text import Text

from mobilerun.cli.base import render
from mobilerun.cli.base.billing import billing_option
from mobilerun.cli.base.run import coro
from mobilerun.clients.cloud import device_flow as deviceauth
from mobilerun.clients.local import login_store as credentials
from mobilerun.config.env import get_secrets, secret_value
from mobilerun.config.settings import get_settings
from mobilerun.models.billing import Billing
from mobilerun.errors.base import MobilerunError


def mask_key(value: str) -> str:
    return f"{value[:3]}…{value[-4:]}" if len(value) > 7 else "…"


@click.command(name="whoami", short_help="Show the current cloud account")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@billing_option
@coro
async def whoami(billing: str, as_json: bool) -> None:
    """Show the currently logged-in Mobilerun cloud user."""
    creds = credentials.load_credentials()
    env_key = secret_value(get_secrets().MOBILERUN_API_KEY)
    if env_key:
        if as_json:
            click.echo(
                json.dumps({"source": "environment", "api_key": mask_key(env_key)})
            )
            return
        render.kv(
            [
                (
                    "API key from environment overrides saved credentials",
                    mask_key(env_key),
                )
            ]
        )
        return
    api_key = (creds or {}).get("api_key")
    if api_key:
        if as_json:
            click.echo(json.dumps({"source": "saved", "api_key": mask_key(api_key)}))
            return
        render.kv([("API key login", mask_key(api_key))])
        return
    token = (creds or {}).get("access_token")
    auth_url = (creds or {}).get("auth_url") or get_settings().MOBILERUN_AUTH_URL
    if not token:
        raise click.ClickException("Not logged in (run `mobilerun account login`).")
    try:
        info = await deviceauth.fetch_session(auth_url, token, billing=Billing(billing))
    except MobilerunError as e:
        raise click.ClickException(f"session check failed: {e}") from e
    if info is None:
        raise click.ClickException(
            "Session not recognized (run `mobilerun account login` again)."
        )
    user = info.get("user", {})
    if as_json:
        click.echo(
            json.dumps(
                {
                    "source": "session",
                    "account": user.get("email", "?"),
                    "user_id": user.get("id", "?"),
                }
            )
        )
        return
    render.kv(
        [
            ("Account", user.get("email", "?")),
            ("User ID", Text(user.get("id", "?"), style="dim")),
        ]
    )


status = whoami
