"""Public provider spellings shared by run and configuration."""

from __future__ import annotations

from dataclasses import dataclass

import click


@dataclass(frozen=True)
class Provider:
    family: str | None
    auth_mode: str
    url_keyword: str
    run_only: str | None = None

    @property
    def runtime(self):
        from mobilerun.agent.providers.registry import resolve_provider_variant

        return (
            resolve_provider_variant(self.family, self.auth_mode).id
            if self.family
            else self.run_only
        )


PROVIDERS = {
    "openai": Provider("openai", "api_key", "api_base"),
    "openai-oauth": Provider("openai", "oauth", "api_base"),
    "anthropic": Provider("anthropic", "api_key", "base_url"),
    "anthropic-oauth": Provider("anthropic", "oauth", "api_base"),
    "gemini": Provider("gemini", "api_key", "http_options"),
    "gemini-oauth": Provider("gemini", "oauth", "code_assist_endpoint"),
    "xai": Provider("xai", "api_key", "api_base"),
    "xai-oauth": Provider("xai", "oauth", "api_base"),
    "ollama": Provider("ollama", "none", "base_url"),
    "openai-like": Provider("openai_like", "api_key", "base_url"),
    "minimax": Provider("minimax", "api_key", "base_url"),
    "zai": Provider("zai", "api_key", "base_url"),
    "openrouter": Provider(None, "api_key", "api_base", "OpenRouter"),
    "deepseek": Provider(None, "api_key", "base_url", "DeepSeek"),
}


_RENAMED_PROVIDERS = {
    "googlegenai": "gemini",
    "google": "gemini",
    "gemini_oauth_code_assist": "gemini-oauth",
    "openairesponses": "openai",
    "gpt": "openai",
    "openai_oauth": "openai-oauth",
    "claude": "anthropic",
    "anthropic_oauth": "anthropic-oauth",
    "xai_oauth": "xai-oauth",
    "openailike": "openai-like",
    "openai_like": "openai-like",
    "openai_compatible": "openai-like",
    "z.ai": "zai",
}


class ProviderChoice(click.Choice):
    """Point a removed provider name at its replacement."""

    def convert(self, value, param, ctx):
        renamed = _RENAMED_PROVIDERS.get(str(value).lower())
        if renamed in self.choices:
            self.fail(
                f"Removed provider name: {value}. Use {renamed} instead.", param, ctx
            )
        return super().convert(value, param, ctx)


def _provider_option(names, note=""):
    return click.option(
        "-p",
        "--provider",
        type=ProviderChoice(names, case_sensitive=False),
        metavar="PROVIDER",
        help="LLM provider: "
        + ", ".join(names)
        + "."
        + (" " + note + "." if note else ""),
    )


provider_option = _provider_option(list(PROVIDERS))
configure_provider_option = _provider_option(
    [name for name, spec in PROVIDERS.items() if spec.family]
)
run_provider_option = _provider_option(
    [*PROVIDERS, "TypeSafe"], note="TypeSafe for the Jev decision model"
)


def llm_url_kwargs(runtime, url):
    if url is None:
        return {}
    # These loaders prefer api_base, so set both or a saved api_base wins.
    if runtime in {"OpenAILike", "DeepSeek", "MiniMax", "ZAI"}:
        return {"base_url": url, "api_base": url}
    spec = next((spec for spec in PROVIDERS.values() if spec.runtime == runtime), None)
    # Grok credentials only ever go to xAI's own endpoint.
    if spec is None or runtime in {"XAI", "xai_oauth"}:
        raise click.UsageError(f"-u/--llm-url is not supported for provider {runtime}")
    return {
        spec.url_keyword: (
            {"base_url": url} if spec.url_keyword == "http_options" else url
        )
    }


def llm_retry_kwargs(runtime):
    """Allow one SDK retry to honour short Retry-After delays.

    The agent's separate retry loop does not yet honour Retry-After.
    """
    from mobilerun.agent.providers.registry import list_provider_families
    from mobilerun.agent.utils.llm_picker import (
        SUPPORTED_PROVIDERS,
        normalize_provider_name,
    )

    runtime = normalize_provider_name(runtime)
    for family in list_provider_families():
        for variant in family.variants:
            if runtime not in {variant.id, variant.runtime_provider_name}:
                continue
            # These adapters have no max_retries constructor field.
            if family.id == "ollama" or (
                variant.auth_mode == "oauth" and family.id in {"anthropic", "gemini"}
            ):
                return {}
            return {"max_retries": 1}
    # Legacy providers outside the family registry still use standard adapters.
    return {"max_retries": 1} if runtime in SUPPORTED_PROVIDERS else {}
