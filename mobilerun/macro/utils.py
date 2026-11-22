"""State matching and logging helpers for macro replay."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from mobilerun.macro.models import node_semantic_key
from mobilerun.macro.settings import get_macro_settings


@dataclass(frozen=True)
class StateMatchResult:
    matches: bool
    score: float
    reason: str


def compare_states(
    saved_state: Dict[str, Any],
    current_state: Dict[str, Any],
    threshold: float | None = None,
) -> StateMatchResult:
    if threshold is None:
        threshold = get_macro_settings().MOBILERUN_MACRO_STATE_THRESHOLD
    saved_keys = _node_key_set(saved_state)
    current_keys = _node_key_set(current_state)

    if saved_keys or current_keys:
        intersection = len(saved_keys & current_keys)
        union = len(saved_keys | current_keys)
        node_score = intersection / union if union else 1.0
    else:
        node_score = 1.0

    phone_score = _phone_score(saved_state, current_state)
    score = round((node_score * 0.85) + (phone_score * 0.15), 4)

    if score >= threshold:
        return StateMatchResult(True, score, f"state matched with score {score:.2f}")

    return StateMatchResult(
        False,
        score,
        f"state similarity {score:.2f} below threshold {threshold:.2f}",
    )


def _node_key_set(state: Dict[str, Any]) -> set[Tuple[Any, ...]]:
    return {
        node_semantic_key(node)
        for node in state.get("nodes", [])
        if any(part is not None and part != "" for part in node_semantic_key(node))
    }


def _phone_score(saved_state: Dict[str, Any], current_state: Dict[str, Any]) -> float:
    saved_phone = saved_state.get("phone_state") or {}
    current_phone = current_state.get("phone_state") or {}
    fields = [
        field
        for field in ("package", "activity")
        if saved_phone.get(field) is not None or current_phone.get(field) is not None
    ]
    if not fields:
        return 1.0
    matches = sum(
        1 for field in fields if saved_phone.get(field) == current_phone.get(field)
    )
    return matches / len(fields)


def configure_macro_logging(debug: bool = False) -> logging.Logger:
    """Configure logging for the macro CLI."""
    logger = logging.getLogger("mobilerun-macro")
    logger.handlers = []

    handler = logging.StreamHandler()

    if debug:
        level = logging.DEBUG
        formatter = logging.Formatter("%(levelname)s %(name)s %(message)s", "%H:%M:%S")
    else:
        level = logging.INFO
        formatter = logging.Formatter("%(message)s", "%H:%M:%S")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    return logger
