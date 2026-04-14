"""
Events for the FastAgent workflow.

Internal events for streaming to frontend/logging.
"""

from typing import Optional

from llama_index.core.workflow import Event

from droidrun.agent.usage import UsageResult


class FastAgentInputEvent(Event):
    """Input ready for LLM."""

    pass


class FastAgentResponseEvent(Event):
    """LLM response received."""

    thought: str
    code: Optional[str] = None
    usage: Optional[UsageResult] = None


class FastAgentToolCallEvent(Event):
    """Tool calls ready to execute."""

    tool_calls_repr: str


class FastAgentOutputEvent(Event):
    """Tool execution result."""

    output: str


class FastAgentEndEvent(Event):
    """FastAgent finished."""

    success: bool
    reason: str
    tool_call_count: int = 0
