"""Public provider spellings shared by run and the configuration wizard."""

from dataclasses import dataclass

import click


@dataclass(frozen=True)
class Provider:
    runtime: str
    family: str
    auth_mode: str
    url_keyword: str


PROVIDERS = {
    "openai": Provider("OpenAIResponses", "openai", "api_key", "api_base"),
    "openai-oauth": Provider("openai_oauth", "openai", "oauth", "api_base"),
    "anthropic": Provider("Anthropic", "anthropic", "api_key", "base_url"),
    "anthropic-oauth": Provider("anthropic_oauth", "anthropic", "oauth", "api_base"),
    "gemini": Provider("GoogleGenAI", "gemini", "api_key", "http_options"),
    "gemini-oauth": Provider("gemini_oauth_code_assist", "gemini", "oauth", "code_assist_endpoint"),
    "xai": Provider("XAI", "xai", "api_key", "api_base"),
    "ollama": Provider("Ollama", "ollama", "none", "base_url"),
    "openai-like": Provider("OpenAILike", "openai_like", "api_key", "api_base"),
    "minimax": Provider("MiniMax", "minimax", "api_key", "api_base"),
    "zai": Provider("ZAI", "zai", "api_key", "api_base"),
}
provider_option = click.option(
    "-p",
    "--provider",
    type=click.Choice(list(PROVIDERS), case_sensitive=False),
    help="LLM provider.",
)


def llm_url_kwargs(runtime, url):
    if url is None:
        return {}
    spec = next(spec for spec in PROVIDERS.values() if spec.runtime == runtime)
    return {spec.url_keyword: {"base_url": url} if spec.url_keyword == "http_options" else url}
