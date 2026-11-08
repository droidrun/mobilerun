"""Host-side approval for sensitive secrets.

A host that wants a human (or any other authority) to confirm the use of a
secret implements :class:`ApprovalProvider` and passes it to ``MobileAgent``.
Authentication, the provenance of the decision and its transport are entirely
the host's responsibility; the engine only asks and obeys the answer.
"""

from abc import ABC, abstractmethod
from typing import Dict, Literal

ApprovalDecision = Literal["approved", "denied", "expired"]


class ApprovalProvider(ABC):
    """Asks the host whether a secret may be typed into the current target."""

    @abstractmethod
    async def request(self, kind: str, details: Dict[str, str]) -> ApprovalDecision:
        """Request approval and wait for the decision.

        Args:
            kind: Host-defined approval kind of the secret
                (``SecretSpec.approval_kind``).
            details: Non-secret context for the approver. Keys: ``label``,
                ``secretGroup``, ``targetPackage`` and, for browsers,
                ``targetHost``. Values are never secret values.

        Returns:
            ``"approved"``, ``"denied"`` or ``"expired"``. Anything else is
            treated as a failure to obtain approval.
        """
