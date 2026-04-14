"""
Abstract base class for app card providers.

Providers load app-specific instruction cards based on package names.
Supports multiple backends: local files, remote servers, or composite strategies.
"""

from abc import ABC, abstractmethod


class AppCardProvider(ABC):
    """Abstract interface for loading app-specific instruction cards."""

    @abstractmethod
    async def load_app_card(self, package_name: str, instruction: str = "") -> str:
        """
        Load app card for a given package asynchronously.

        Args:
            package_name: Android package name (e.g., "com.google.android.gm")
            instruction: User's instruction/goal (optional context for server providers)

        Returns:
            App card content as string, or empty string if not found or on error
        """
        pass
