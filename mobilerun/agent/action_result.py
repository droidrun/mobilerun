"""ActionResult — structured return type from action functions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionResult:
    """What the agent sees after an action runs."""

    success: bool
    summary: str

    def __str__(self) -> str:
        return self.summary
