"""Shared Grok/xAI model and transport metadata."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

# grok-4.7 on the public API does not emit mobilerun's XML tool calls; the
# Grok Build proxy serves a build that does.
GROK_API_DEFAULT_MODEL = "grok-4.6"
GROK_OAUTH_DEFAULT_MODEL = "grok-4.7"
GROK_API_MODELS = (GROK_API_DEFAULT_MODEL,)
GROK_OAUTH_MODELS = (GROK_OAUTH_DEFAULT_MODEL, "grok-4.6")
# No longer offered, but still accepted so saved OAuth profiles load.
GROK_OAUTH_LEGACY_MODELS = ("grok-4.5",)
GROK_MODEL_ALIASES = {
    "grok-4.5-latest": "grok-4.5",
}

XAI_API_BASE = "https://api.x.ai/v1"
GROK_CONTEXT_WINDOW = 500_000

# Chat Completions-only controls that mobilerun keeps out of Grok Responses
# payloads. Filter after all constructor and per-call kwargs are merged so an
# override cannot accidentally restore them.
GROK_UNSUPPORTED_SAMPLING_PARAMS = frozenset(
    {"presence_penalty", "frequency_penalty", "stop"}
)


def normalize_grok_model_id(model: object) -> str:
    """Normalize public xAI/Grok aliases to Mobilerun's canonical model id."""

    model_id = str(model or "").strip()
    if model_id.startswith("xai/"):
        model_id = model_id.removeprefix("xai/")
    return GROK_MODEL_ALIASES.get(model_id, model_id)


def sanitize_grok_responses_kwargs(
    payload: MutableMapping[str, Any],
    *,
    omit_sampler_fields: bool = False,
    omit_tool_choice: bool = False,
) -> MutableMapping[str, Any]:
    """Apply Grok's Responses API parameter contract to a final payload."""

    filtered_params = set(GROK_UNSUPPORTED_SAMPLING_PARAMS)
    if omit_sampler_fields:
        filtered_params.update(("temperature", "top_p"))
    if omit_tool_choice:
        filtered_params.add("tool_choice")

    for param in filtered_params:
        payload.pop(param, None)
    payload["store"] = False
    payload.pop("reasoning", None)

    # The OpenAI SDK merges ``extra_body`` after its normal typed parameters,
    # so an unsanitized value here could otherwise restore storage, reasoning,
    # an unsupported sampler, or even a caller-selected model. Preserve other
    # extension fields while removing every value Mobilerun pins or rejects.
    extra_body = payload.get("extra_body")
    if extra_body is not None:
        if not isinstance(extra_body, Mapping):
            payload.pop("extra_body", None)
        else:
            sanitized_extra_body = dict(extra_body)
            for param in filtered_params:
                sanitized_extra_body.pop(param, None)
            for param in ("model", "store", "reasoning"):
                sanitized_extra_body.pop(param, None)
            payload["extra_body"] = sanitized_extra_body
    return payload
