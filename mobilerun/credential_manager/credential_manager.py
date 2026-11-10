from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal


class CredentialNotFoundError(KeyError):
    """Raised when a credential key is not found."""

    pass


@dataclass(frozen=True)
class SecretSpec:
    """Non-secret metadata about one credential key."""

    key: str
    # Typed only after the host's ApprovalProvider approves.
    requires_approval: bool = False
    # Host-defined approval category, passed to ``ApprovalProvider.request``.
    approval_kind: str | None = None
    # Human-readable name shown to the approver. Never a secret value.
    approval_label: str | None = None
    # "global": mask the value everywhere in observations.
    # "input_only": mask it only in focused/editable nodes (short values such
    # as a verification code that may legitimately appear elsewhere on screen).
    redact: Literal["global", "input_only"] = "global"
    # Resolve and mask before the run's first observation; typing still needs approval.
    mask_from_start: bool = False
    # Mask every Luhn-valid card number in this run, including undeclared cards.
    # Luhn also matches IMEI and ICCID identifiers.
    mask_card_numbers: bool = False


class CredentialManager(ABC):
    """Abstract base class for credential resolution."""

    @abstractmethod
    async def resolve_key(self, key: str) -> str:
        """
        Resolve and return the value for the given credential key.

        Args:
            key: Credential identifier

        Returns:
            The credential value as a string

        Raises:
            CredentialNotFoundError: If key doesn't exist
        """
        pass

    @abstractmethod
    async def get_keys(self) -> List[str]:
        """
        Get all available credential keys.

        Returns:
            List of credential identifiers
        """
        pass

    async def get_secret_specs(self) -> List[SecretSpec]:
        """Get metadata for all available credential keys.

        The default describes every key as an ordinary secret, so existing
        subclasses keep working unchanged. Override to mark secrets as
        approval-gated.
        """
        return [SecretSpec(key=key) for key in await self.get_keys()]
