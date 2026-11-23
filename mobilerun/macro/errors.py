"""A guarded macro stopped before an action could safely execute."""

from typing import Any

from mobilerun.errors.base import MobilerunError


class MacroError(MobilerunError):
    """Base error for macro operations."""


class MacroMismatch(MacroError, RuntimeError):
    def __init__(
        self,
        step_index: int,
        remaining_actions: list[dict[str, Any]],
        current_state: dict[str, Any],
        reason: str = "state mismatch",
    ) -> None:
        self.step_index = step_index
        self.remaining_actions = remaining_actions
        self.current_state = current_state
        self.reason = reason
        super().__init__(f"Macro state mismatch before step {step_index + 1}: {reason}")
