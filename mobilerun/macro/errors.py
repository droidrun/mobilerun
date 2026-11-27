"""Errors for building and playing deterministic macros."""

from typing import Literal

from mobilerun.errors.base import MobilerunError
from mobilerun.macro.models import MacroStep
from mobilerun.models import Fingerprint

MismatchPhase = Literal["before", "action", "after"]


class MacroError(MobilerunError):
    """Base error for macro operations."""


class MacroMismatch(MacroError, RuntimeError):
    """Replay stopped at a guard or action for an original macro row."""

    value_safe = True

    def __init__(
        self,
        step: MacroStep,
        phase: MismatchPhase,
        observed: Fingerprint | None,
    ) -> None:
        self.step = step
        self.phase = phase
        self.observed = observed
        super().__init__(f"Macro mismatch at step {step.i} ({phase})")
