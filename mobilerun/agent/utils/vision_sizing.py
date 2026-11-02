"""Resolve the exact screenshot size a vision model actually grounds on.

Some providers downsize images server-side before the model sees them (e.g.
Anthropic's 28px visual-token budget). If the coordinate contract declares a
larger space than the model receives, the model grounds on the smaller image it
actually sees and coordinate actions land short. This policy computes the EXACT
dimensions the model will use, so the declared space == the sent image and
``convert_point`` stays exact.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from mobilerun.agent.providers.anthropic import ANTHROPIC_HIGHRES_MODELS
from mobilerun.tools.helpers.images import (
    MODEL_SCREENSHOT_MAX_SIDE,
    anthropic_resized_size,
    fit_dimensions_to_max_side,
)

# Anthropic models with the high-resolution budget (2576 px / 4784 tokens).
# Unknown Anthropic ids fall back to the standard 1568 budget — never assume
# high-res, since that would re-introduce the undershoot bug.
_ANTHROPIC_STANDARD = (1568, 1568)  # (max_edge, max_tokens)
_ANTHROPIC_HIGHRES = (2576, 4784)

# Models that ground poorly past this long edge, even though the provider
# accepts a larger image.
_GROUNDING_MAX_SIDE = {"claude-sonnet-5": 1568}


def _model_id(llm: Any) -> str:
    return str(getattr(llm, "model", "") or "")


def _is_anthropic(model_id: str) -> bool:
    return model_id.startswith("claude")


def model_uses_normalized_coordinates(model_id: str) -> bool:
    """Whether the model answers screenshot positions on a 0-1000 grid.

    Gemma and Gemini 3.8 Flash ground in normalized coordinates and ignore a
    declared pixel space.
    """
    model = model_id.lower()
    return "gemma" in model or "gemini-3.8-flash" in model


def model_effective_dims(model_id: str, width: int, height: int) -> tuple[int, int]:
    """Exact dims a single model grounds on for a native ``width``x``height`` screen."""
    base_w, base_h = fit_dimensions_to_max_side(
        width, height, MODEL_SCREENSHOT_MAX_SIDE
    )
    if _is_anthropic(model_id):
        edge, tokens = (
            _ANTHROPIC_HIGHRES
            if model_id in ANTHROPIC_HIGHRES_MODELS
            else _ANTHROPIC_STANDARD
        )
        base_w, base_h = anthropic_resized_size(base_w, base_h, edge, tokens)
    # OpenAI / Gemini / Ollama / OpenAI-compatible: ground at the declared size
    # (empirically no further server-side downsize for the supported models).
    limit = _GROUNDING_MAX_SIDE.get(model_id)
    if limit and max(base_w, base_h) > limit:
        base_w, base_h = fit_dimensions_to_max_side(base_w, base_h, limit)
    return base_w, base_h


class VisionResizePolicy:
    """Composite resize policy across all active vision models.

    Resolves the screenshot dimensions the contract should declare and send so
    the image matches what EVERY screenshot recipient grounds on. Uses the
    smallest (most conservative) result across models, then an optional explicit
    max-side cap (escape hatch for undocumented local/Ollama models).
    """

    def __init__(
        self, model_ids: Sequence[str], max_side_cap: Optional[int] = None
    ) -> None:
        self._model_ids = [m for m in dict.fromkeys(model_ids) if m]
        self._cap = max_side_cap if (max_side_cap and max_side_cap > 0) else None

    @classmethod
    def from_llms(
        cls, llms: Sequence[Any], max_side_cap: Optional[int] = None
    ) -> "VisionResizePolicy":
        return cls(
            [_model_id(llm) for llm in llms if llm is not None],
            max_side_cap=max_side_cap,
        )

    def effective_dims(self, width: int, height: int) -> tuple[int, int]:
        """Smallest effective dims across all models, after the optional cap."""
        if width <= 0 or height <= 0:
            return width, height
        candidates = [
            model_effective_dims(m, width, height) for m in self._model_ids
        ] or [fit_dimensions_to_max_side(width, height, MODEL_SCREENSHOT_MAX_SIDE)]
        # Aspect ratio is preserved, so smallest long-edge == smallest area.
        w, h = min(candidates, key=max)
        if self._cap and max(w, h) > self._cap:
            w, h = fit_dimensions_to_max_side(w, h, self._cap)
        return w, h
