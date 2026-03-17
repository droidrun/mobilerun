"""
Events for the CodeActAgent workflow.

Internal events for streaming to frontend/logging.
"""

from typing import Optional

from llama_index.core.workflow import Event

from droidrun.agent.usage import UsageResult


class CodeActInputEvent(Event):
    """Input ready for LLM."""

    pass


class CodeActResponseEvent(Event):
    """LLM response received."""

    thought: str
    code: Optional[str] = None
    usage: Optional[UsageResult] = None


class CodeActCodeEvent(Event):
    """Code ready to execute (internal event)."""

    code: str


class CodeActOutputEvent(Event):
    """Code execution result (internal event)."""

    output: str


class CodeActEndEvent(Event):
    """CodeAct finished."""

    success: bool
    reason: str
    code_executions: int = 0


# ============================================================================
# FastAgent events (used by ToolsAgent / FastAgent)
# ============================================================================


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
    images: list[bytes] = []


class FastAgentEndEvent(Event):
    """FastAgent finished."""

    success: bool
    reason: str
    tool_call_count: int = 0
