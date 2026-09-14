"""
Events for the ManagerAgent workflow.

Internal events for streaming to frontend/logging.
For MobileAgent coordination events, see droid/events.py
"""

from typing import Any, Optional

from llama_index.core.workflow import Event

from mobilerun.agent.usage import UsageResult


class ManagerContextEvent(Event):
    """Context prepared, ready for LLM call."""

    pass


class ManagerResponseEvent(Event):
    """LLM response received, ready for parsing."""

    response: str
    usage: Optional[UsageResult] = None
    parsed_response: Optional[dict[str, Any]] = None
    structured_output_mode: Optional[str] = None
    semantic_retries: int = 0


class ManagerPlanDetailsEvent(Event):
    """Plan parsed and ready (internal event with full details)."""

    plan: str
    subgoal: str
    thought: str
    answer: str = ""
    memory_update: str = ""
    progress_summary: str = ""
    success: Optional[bool] = None  # True/False if complete, None if in progress
    full_response: str = ""
