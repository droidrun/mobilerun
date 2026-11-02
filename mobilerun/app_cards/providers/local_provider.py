"""
Local file-based app card provider.

Loads app cards from one folder per platform, each with its own
app_cards.json mapping.
"""

import json
import logging
from pathlib import Path
from typing import Dict

from mobilerun.app_cards.app_card_provider import AppCardProvider
from mobilerun.config_manager.path_resolver import PathResolver

logger = logging.getLogger("mobilerun")

PLATFORMS = ("android", "ios")
_MAPPING_FILE = "app_cards.json"


def _resolve_cards_folder(app_cards_dir: str) -> Path:
    """Use the working-dir folder if it has app cards, else the package one."""
    path = Path(app_cards_dir).expanduser()
    if path.is_absolute():
        _warn_if_flat(path)
        return path
    candidates = (Path.cwd() / path, PathResolver.get_project_root() / path)
    for folder in candidates:
        _warn_if_flat(folder)
        if any((folder / p / _MAPPING_FILE).exists() for p in PLATFORMS):
            return folder
    return candidates[0]


def _warn_if_flat(folder: Path) -> None:
    flat = folder / _MAPPING_FILE
    if flat.exists():
        logger.warning(
            f"Ignoring {flat}: app cards are read from {folder / 'android'} "
            f"and {folder / 'ios'}"
        )


class LocalAppCardProvider(AppCardProvider):
    """Load app cards from local filesystem with in-memory caching.

    Layout: <app_cards_dir>/android/app_cards.json and
    <app_cards_dir>/ios/app_cards.json, each mapping an app id to a card
    file in the same folder.
    """

    def __init__(self, app_cards_dir: str = "config/app_cards"):
        """
        Initialize local provider.

        Args:
            app_cards_dir: Directory containing the android/ and ios/ card folders
        """
        self.app_cards_dir = _resolve_cards_folder(app_cards_dir)

        # platform -> (cards folder, mapping)
        self._mappings: Dict[str, tuple[Path, Dict[str, str]]] = {}
        for platform in PLATFORMS:
            relative = f"{platform}/{_MAPPING_FILE}"
            mapping_path = self.app_cards_dir / relative
            if not mapping_path.exists():
                continue
            try:
                with open(mapping_path, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {mapping_path}: {e}")
                continue
            if not isinstance(mapping, dict):
                logger.warning(f"Ignoring {mapping_path}: expected a JSON object")
                continue
            self._mappings[platform] = (mapping_path.parent, mapping)
            logger.debug(f"Loaded {relative} with {len(mapping)} entries")

        if not self._mappings:
            logger.warning(
                f"No android/ or ios/ app_cards.json found in {self.app_cards_dir}"
            )

        # Content cache: (package_name, instruction, platform) -> content
        self._content_cache: Dict[tuple[str, str, str | None], str] = {}

    async def load_app_card(
        self, package_name: str, instruction: str = "", platform: str | None = None
    ) -> str:
        """
        Load app card for a package name from local files.

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

        source = self._mappings.get(platform or "")
        if source is None or package_name not in source[1]:
            self._content_cache[cache_key] = ""
            return ""

        # Get app card file path (relative to the platform folder)
        cards_folder, mapping = source
        filename = mapping[package_name]
        if not isinstance(filename, str):
            self._content_cache[cache_key] = ""
            logger.warning(f"Invalid app card path for {package_name}: {filename!r}")
            return ""
        app_card_path = cards_folder / filename

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
            "mapping_entries": sum(len(m) for _, m in self._mappings.values()),
        }
