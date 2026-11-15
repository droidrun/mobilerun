"""State matching for guarded macro replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from mobilerun.macro.state import OPTIONAL_SEMANTIC_FIELDS, node_semantic_key


@dataclass(frozen=True)
class StateMatchResult:
    matches: bool
    score: float
    reason: str


def compare_states(
    saved_state: Dict[str, Any],
    current_state: Dict[str, Any],
    threshold: float = 0.85,
) -> StateMatchResult:
    # Skip flags a snapshot never recorded (older macros, uiautomator states).
    skip = _unrecorded_fields(saved_state) | _unrecorded_fields(current_state)
    saved_keys = _node_key_set(saved_state, skip)
    current_keys = _node_key_set(current_state, skip)

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


def _unrecorded_fields(state: Dict[str, Any]) -> set[str]:
    nodes = state.get("nodes", [])
    return {
        field
        for field in OPTIONAL_SEMANTIC_FIELDS
        if all(node.get(field) is None for node in nodes)
    }


def _node_key_set(
    state: Dict[str, Any], skip: set[str] = frozenset()
) -> set[Tuple[Any, ...]]:
    keys = (node_semantic_key(node, skip) for node in state.get("nodes", []))
    return {key for key in keys if any(part is not None and part != "" for part in key)}


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
