from abc import ABC, abstractmethod
from typing import List


class CredentialNotFoundError(KeyError):
    """Raised when a credential key is not found."""

    pass


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
