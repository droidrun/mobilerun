"""Public provider spellings shared by run and configuration."""

from dataclasses import dataclass

import click

from mobilerun.agent.providers.registry import resolve_provider_variant


@dataclass(frozen=True)
class Provider:
    family: str | None
    auth_mode: str
    url_keyword: str
    run_only: str | None = None

    @property
    def runtime(self):
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
    "ollama": Provider("ollama", "none", "base_url"),
    "openai-like": Provider("openai_like", "api_key", "base_url"),
    "minimax": Provider("minimax", "api_key", "base_url"),
    "zai": Provider("zai", "api_key", "base_url"),
    "openrouter": Provider(None, "api_key", "api_base", "OpenRouter"),
    "deepseek": Provider(None, "api_key", "base_url", "DeepSeek"),
}


def _provider_option(names, note=""):
    return click.option(
        "-p",
        "--provider",
        type=click.Choice(names, case_sensitive=False),
        metavar="PROVIDER",
        help="LLM provider: "
        + ", ".join(names)
        + "."
        + (" " + note + "." if note else ""),
    )


provider_option = _provider_option(list(PROVIDERS))
run_provider_option = _provider_option(
    [*PROVIDERS, "TypeSafe"], note="TypeSafe for the Jev decision model"
)


def llm_url_kwargs(runtime, url):
    if url is None:
        return {}
    spec = next((spec for spec in PROVIDERS.values() if spec.runtime == runtime), None)
    if spec is None:
        # Only these loaders normalise the generic base_url keyword.
        if runtime in {"OpenAILike", "DeepSeek", "MiniMax", "ZAI"}:
            return {"base_url": url}
        raise click.UsageError(f"-u/--llm-url is not supported for provider {runtime}")
    if runtime == "XAI":
        raise click.UsageError("-u/--llm-url is not supported for provider XAI")
    return {
        spec.url_keyword: (
            {"base_url": url} if spec.url_keyword == "http_options" else url
        )
    }
