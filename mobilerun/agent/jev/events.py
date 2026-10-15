"""Workflow events emitted by the Jev agent. They never carry secrets."""

from typing import Optional

from llama_index.core.workflow import Event


class JevDecisionEvent(Event):
    """Jev answered one step's questions."""

    step: int
    status: str
    operation: str
    confidence: float
    label: Optional[str] = None
    target_confidence: Optional[float] = None
    model: Optional[str] = None
    latency_ms: int = 0
    input_tokens: Optional[int] = None


class JevStepEvent(Event):
    """An action chosen by Jev finished executing."""

    step: int
    operation: str
    success: bool
    summary: str


class JevEndEvent(Event):
    """The Jev loop stopped."""

    success: bool
    reason: str
    steps: int
