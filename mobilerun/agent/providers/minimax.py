from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from llama_index.core.types import PydanticProgramMode

MINIMAX_GLOBAL_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_CHINA_BASE_URL = "https://api.minimaxi.com/v1"
MINIMAX_LEGACY_BASE_URL = "https://api.minimaxi.chat/v1"
MINIMAX_DEFAULT_MODEL = "MiniMax-M3"

MINIMAX_MODEL_CONTEXT_WINDOWS = {
    "MiniMax-M3": 1_000_000,
    "MiniMax-M2.7": 204_800,
    "MiniMax-M2.7-highspeed": 204_800,
    "MiniMax-M2.5": 204_800,
    "MiniMax-M2.5-highspeed": 204_800,
    "MiniMax-M2.1": 204_800,
    "MiniMax-M2.1-highspeed": 204_800,
    "MiniMax-M2": 204_800,
}

logger = logging.getLogger("mobilerun")


def _normalize_base_url(base_url: str | None) -> str:
    return str(base_url or "").strip().rstrip("/").lower()


def minimax_model_context_window(model: object) -> int | None:
    """Return the documented context window for a known MiniMax model."""
    return MINIMAX_MODEL_CONTEXT_WINDOWS.get(str(model or "").strip())


def is_minimax_base_url(base_url: str | None) -> bool:
    return _normalize_base_url(base_url) in {
        _normalize_base_url(url)
        for url in (
            MINIMAX_GLOBAL_BASE_URL,
            MINIMAX_CHINA_BASE_URL,
            MINIMAX_LEGACY_BASE_URL,
        )
    }


def apply_minimax_defaults(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Fill MiniMax capabilities that OpenAILike cannot infer."""
    kwargs = dict(kwargs)
    kwargs.setdefault("is_function_calling_model", True)
    # MiniMax treats forced tool_choice as a hint, so structured output uses
    # the text program.
    kwargs.setdefault("pydantic_program_mode", PydanticProgramMode.LLM)
    context_window = minimax_model_context_window(kwargs.get("model"))
    if context_window is not None:
        kwargs.setdefault("context_window", context_window)
    # Return thinking in reasoning_details instead of inline <think> text.
    additional_kwargs = dict(kwargs.get("additional_kwargs") or {})
    extra_body = dict(additional_kwargs.get("extra_body") or {})
    extra_body.setdefault("reasoning_split", True)
    additional_kwargs["extra_body"] = extra_body
    kwargs["additional_kwargs"] = additional_kwargs
    return kwargs


@lru_cache(maxsize=1)
def _warn_about_legacy_endpoint_once() -> None:
    logger.warning(
        "This MiniMax profile uses the legacy endpoint "
        f"{MINIMAX_LEGACY_BASE_URL}. The profile was not changed. Re-run "
        "`mobilerun configure` and choose either Global "
        f"({MINIMAX_GLOBAL_BASE_URL}) or Mainland China "
        f"({MINIMAX_CHINA_BASE_URL})."
    )


def warn_if_legacy_minimax_endpoint(base_url: str | None) -> None:
    """Warn once per process when a profile still uses MiniMax's legacy endpoint."""
    if _normalize_base_url(base_url) == _normalize_base_url(MINIMAX_LEGACY_BASE_URL):
        _warn_about_legacy_endpoint_once()
