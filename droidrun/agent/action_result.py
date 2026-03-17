"""ActionResult — structured return type from action functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TypedDict


@dataclass
class ActionResult:
    """What the agent sees after an action runs."""

    success: bool
    summary: str
    image: Optional[bytes] = field(default=None, repr=False)

    def __str__(self) -> str:
        return self.summary


class ActionRecord(TypedDict):
    """Record of an executed tool call, used across Executor/DroidAgent/Manager."""

    action: str
    args: Dict[str, Any]
    outcome: bool
    error: str
    summary: str
