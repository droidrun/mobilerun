"""Interactive configuration command."""

import click
from rich.console import Console

from mobilerun.cli.auth import (
    _run_anthropic_oauth_login,
    _run_gemini_oauth_login,
    _run_grok_oauth_login,
    _run_openai_oauth_login,
)
from mobilerun.cli.configure_wizard import (
    ConfigureWizardCallbacks,
    _oauth_credentials_present,
    run_configure_wizard,
)
from mobilerun.cli.options import (
    CONTEXT_SETTINGS,
    Command,
    config_option,
    input_allowed,
    llm_url_option,
    model_option,
    no_input_option,
    read_secret,
    verbose_option,
)
from mobilerun.cli.providers import PROVIDERS, configure_provider_option


@click.command(
    cls=Command, context_settings=CONTEXT_SETTINGS,
    epilog='\b\nExamples:\n  mobilerun configure\n  mobilerun configure -p openai -m MODEL',
)
@configure_provider_option
@model_option
@llm_url_option
@verbose_option
@config_option
@no_input_option
@click.option(
    "-k", "--api-key", callback=read_secret,
    help="Provider API key; prefer '-' for stdin or the provider's environment variable, "
         "because flag values end up in shell history.",
)
@click.option("--auth-mode", type=click.Choice(["api_key", "oauth", "none"]), help="Authentication mode for the provider.")
def configure(provider, model, llm_url, verbose, config_path, api_key, auth_mode, no_input):
    """Configure provider, authentication and model with the wizard."""
    spec = PROVIDERS.get(provider)
    if spec:
        if spec.family is None:
            raise click.UsageError(f"Provider {provider} is supported by run only")
        auth_mode = auth_mode or spec.auth_mode
    if not input_allowed(no_input):
        _validate_non_interactive(provider, model, api_key, llm_url, auth_mode)
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


def _validate_non_interactive(provider, model, api_key, base_url, auth_mode):
    """Validate before loading or saving config, or entering the wizard."""
    from mobilerun.agent.providers.registry import VARIANT_ENV_KEY_SLOT, resolve_provider_variant
    from mobilerun.config_manager.env_keys import resolve_env_key

    if not provider:
        raise click.UsageError("Missing option '-p / --provider'; specify a provider.")
    if not model:
        raise click.UsageError("Missing option '-m / --model'; specify a model.")
    spec = PROVIDERS[provider]
    try:
        variant = resolve_provider_variant(spec.family, auth_mode)
    except ValueError as exc:
        raise click.UsageError(f"{exc}; check --auth-mode for -p {provider}.") from None
    slot = VARIANT_ENV_KEY_SLOT.get(variant.id)
    if variant.requires_api_key and not (api_key or (slot and resolve_env_key(slot, "auto"))):
        raise click.UsageError(
            "Missing option '-k / --api-key'; pass -k - to read stdin, or set the provider's environment variable."
        )
    if variant.requires_base_url and not (base_url or variant.base_url):
        raise click.UsageError("Missing option '-u / --llm-url'; specify the provider endpoint.")
    if variant.auth_mode == "oauth" and not _oauth_credentials_present(variant.credential_path, variant.id):
        raise click.UsageError(
            f"Saved OAuth credentials required; run `mobilerun auth login {spec.family}` "
            "in a terminal first, or choose -p PROVIDER with API-key authentication."
        )
