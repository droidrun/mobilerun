"""Base interface for accessibility tree filters."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class TreeFilter(ABC):
    """Interface for filtering accessibility trees."""

    @abstractmethod
    def filter(
        self, a11y_tree: Dict[str, Any], device_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Filter tree and return filtered tree with hierarchy preserved."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return filter name."""
        pass
