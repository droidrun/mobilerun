"""
Local file-based app card provider.

Loads app cards from local filesystem using app_cards.json mapping.
"""

import json
import logging
from typing import Dict

from mobilerun.app_cards.app_card_provider import AppCardProvider
from mobilerun.config_manager.path_resolver import PathResolver

logger = logging.getLogger("mobilerun")


class LocalAppCardProvider(AppCardProvider):
    """Load app cards from local filesystem with in-memory caching."""

    def __init__(self, app_cards_dir: str = "config/app_cards"):
        """
        Initialize local provider.

        Args:
            app_cards_dir: Directory containing app_cards.json and markdown files
        """
        # Resolve app_cards.json path once
        mapping_path = PathResolver.resolve(f"{app_cards_dir}/app_cards.json")
        self.app_cards_dir = mapping_path.parent

        # Load mapping immediately
        try:
            if mapping_path.exists():
                with open(mapping_path, "r", encoding="utf-8") as f:
                    self.mapping = json.load(f)
                logger.debug(f"Loaded app_cards.json with {len(self.mapping)} entries")
            else:
                logger.warning(f"app_cards.json not found at {mapping_path}")
                self.mapping = {}
        except Exception as e:
            logger.warning(f"Failed to load app_cards.json: {e}")
            self.mapping = {}

        # Content cache: (package_name, instruction, platform) -> content
        self._content_cache: Dict[tuple[str, str, str | None], str] = {}

    async def load_app_card(
        self, package_name: str, instruction: str = "", platform: str | None = None
    ) -> str:
        """
        Load app card for a package name from local files.

        A mapping value is either a card path for every platform, or a dict of
        per-platform paths with an optional "default", e.g.
        {"android": "android/tiktok.md", "ios": "ios/tiktok.md"}.

        Args:
            package_name: Android package name or iOS bundle id
            instruction: User instruction (for cache key consistency, not used in loading)
            platform: "android" or "ios", when known

        Returns:
            App card content or empty string if not found
        """
        if not package_name:
            return ""

        # Check content cache first
        cache_key = (package_name, instruction, platform)
        if cache_key in self._content_cache:
            logger.debug(f"App card cache hit: {package_name}")
            return self._content_cache[cache_key]

        # Check if package exists in mapping
        if package_name not in self.mapping:
            self._content_cache[cache_key] = ""
            return ""

        # Get app card file path (relative to app_cards_dir)
        filename = self.mapping[package_name]
        if isinstance(filename, dict):
            filename = filename.get(platform or "") or filename.get("default")
        if not filename:
            self._content_cache[cache_key] = ""
            logger.debug(f"No app card for {package_name} on {platform}")
            return ""
        if not isinstance(filename, str):
            self._content_cache[cache_key] = ""
            logger.warning(f"Invalid app card path for {package_name}: {filename!r}")
            return ""
        app_card_path = self.app_cards_dir / filename

        # Read file
        try:
            if not app_card_path.exists():
                self._content_cache[cache_key] = ""
                logger.debug(f"App card not found: {app_card_path}")
                return ""

            # Async file read
            import asyncio

            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(None, app_card_path.read_text, "utf-8")

            # Cache and return
            self._content_cache[cache_key] = content
            logger.debug(f"Loaded app card for {package_name} from {app_card_path}")
            return content

        except Exception as e:
            logger.warning(f"Failed to load app card for {package_name}: {e}")
            self._content_cache[cache_key] = ""
            return ""

    def clear_cache(self) -> None:
        """Clear content cache (useful for testing or runtime reloading)."""
        self._content_cache.clear()
        logger.debug("Local app card cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (useful for debugging)
        """
        return {
            "content_entries": len(self._content_cache),
            "mapping_entries": len(self.mapping),
        }
