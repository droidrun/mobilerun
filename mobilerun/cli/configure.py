"""Interactive configuration command."""

import click
from rich.console import Console

from mobilerun.cli.auth import (
    _run_anthropic_oauth_login,
    _run_gemini_oauth_login,
    _run_grok_oauth_login,
    _run_openai_oauth_login,
)
from mobilerun.cli.configure_wizard import ConfigureWizardCallbacks, run_configure_wizard
from mobilerun.cli.options import (
    CONTEXT_SETTINGS,
    Command,
    config_option,
    llm_url_option,
    model_option,
    verbose_option,
)
from mobilerun.cli.providers import PROVIDERS, provider_option


@click.command(cls=Command, context_settings=CONTEXT_SETTINGS)
@provider_option
@model_option
@llm_url_option
@verbose_option
@config_option
@click.option("-k", "--api-key", help="API key for the selected provider.")
@click.option("--auth-mode", type=click.Choice(["api_key", "oauth", "none"]), help="Authentication mode for the provider.")
def configure(provider, model, llm_url, verbose, config_path, api_key, auth_mode):
    """Configure provider, authentication and model with the wizard."""
    spec = PROVIDERS.get(provider)
    if spec:
        if spec.family is None:
            raise click.UsageError(f"Provider {provider} is supported by run only")
        auth_mode = auth_mode or spec.auth_mode
    run_configure_wizard(
        Console(stderr=True),
        ConfigureWizardCallbacks(
            run_openai_oauth_login=_run_openai_oauth_login,
            run_anthropic_oauth_login=_run_anthropic_oauth_login,
            run_gemini_oauth_login=_run_gemini_oauth_login,
            run_grok_oauth_login=_run_grok_oauth_login,
        ),
        provider=spec.family if spec else None,
        auth_mode=auth_mode,
        model=model,
        api_key=api_key,
        base_url=llm_url,
        config_path=config_path,
    )
