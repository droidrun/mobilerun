"""
Server-based app card provider.

Fetches app cards from a remote HTTP server.
"""

import logging
from typing import Dict

import httpx

from droidrun.app_cards.app_card_provider import AppCardProvider

logger = logging.getLogger("droidrun")


class ServerAppCardProvider(AppCardProvider):
    """Load app cards from remote server with in-memory caching."""

    def __init__(self, server_url: str, timeout: float = 2.0, max_retries: int = 2):
        """
        Initialize server provider.

        Args:
            server_url: Base URL of the app card server (e.g., "https://api.example.com")
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts on failure
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._content_cache: Dict[tuple[str, str], str] = {}

    async def load_app_card(self, package_name: str, instruction: str = "") -> str:
        """
        Load app card from remote server.

        Args:
            package_name: Android package name (e.g., "com.google.android.gm")
            instruction: User instruction/goal (sent to server for context)

        Returns:
            App card content or empty string if not found or on error
        """
        if not package_name:
            return ""

        # Check content cache first (key: package_name, instruction)
        cache_key = (package_name, instruction)
        if cache_key in self._content_cache:
            return self._content_cache[cache_key]

        # Make HTTP request with retries
        endpoint = f"{self.server_url}/app-cards"
        payload = {"package_name": package_name, "instruction": instruction}

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(endpoint, json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        app_card = data.get("app_card", "")

                        # Cache the result (even if empty)
                        self._content_cache[cache_key] = app_card
                        return app_card

                    elif response.status_code == 404:
                        # Not found is expected, cache empty result
                        self._content_cache[cache_key] = ""
                        return ""

                    else:
                        logger.warning(
                            f"Server returned status {response.status_code} for {package_name} "
                            f"(attempt {attempt}/{self.max_retries})"
                        )

            except httpx.TimeoutException:
                logger.warning(
                    f"Server request timeout for {package_name} "
                    f"(attempt {attempt}/{self.max_retries})"
                )

            except httpx.RequestError as e:
                logger.warning(
                    f"Server request failed for {package_name}: {e} "
                    f"(attempt {attempt}/{self.max_retries})"
                )

            except Exception as e:
                logger.warning(
                    f"Unexpected error loading app card from server: {e} "
                    f"(attempt {attempt}/{self.max_retries})"
                )

        # All retries failed, cache empty result
        logger.warning(
            f"Failed to load app card from server after {self.max_retries} attempts"
        )
        self._content_cache[cache_key] = ""
        return ""

    def clear_cache(self) -> None:
        """Clear content cache."""
        self._content_cache.clear()
        logger.debug("Server app card cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (useful for debugging)
        """
        return {
            "content_entries": len(self._content_cache),
        }
