"""
Unified path resolution for Droidrun.

This module provides a single path resolver that handles all file path resolution
with consistent priority: working directory first, then package directory.
"""

from pathlib import Path
from typing import Union


class PathResolver:
    """
    Unified path resolver for all Droidrun file operations.

    Resolution order:
    1. Absolute paths → use as-is
    2. Relative paths → check working dir first, then package dir (bundled resources)
    3. For creation → prefer working dir
    """

    @staticmethod
    def get_project_root() -> Path:
        """
        Get the package root directory (where config/ bundled resources live).

        This is 1 parent up from this file's location:
        droidrun/config_manager/path_resolver.py -> droidrun/ (package root)
        """
        return Path(__file__).resolve().parents[1]

    @staticmethod
    def resolve(
        path: Union[str, Path],
        create_if_missing: bool = False,
        must_exist: bool = False,
    ) -> Path:
        """
        Universal path resolver for all file operations.

        Resolution order:
        1. Absolute path → use as-is
        2. Relative path:
           - If creating: prefer working directory (for user outputs)
           - If reading: check working dir first, then package dir (for bundled resources)
        3. If must_exist and not found → raise FileNotFoundError

        Args:
            path: Path to resolve (str or Path object). Use absolute paths to avoid
                 ambiguity, or relative paths for portable configs.
            create_if_missing: If True, prefer working dir for relative paths (output mode).
                              Does not create the file/directory, just determines the location.
            must_exist: If True, raise FileNotFoundError if path doesn't exist after resolution.
                       Use for required config files and prompts.

        Returns:
            Resolved absolute Path object

        Raises:
            FileNotFoundError: If must_exist=True and path not found in any location.
                              Error message includes both checked locations.

        Security Note:
            Expands tilde (~) to user home directory. Does not follow symlinks during
            resolution, but the returned path may point to a symlink target.

        Examples:
            # Reading config (checks CWD first, then package dir)
            config_path = PathResolver.resolve("config.yaml")

            # Creating output (creates in CWD)
            output_dir = PathResolver.resolve("trajectories", create_if_missing=True)

            # Loading prompts (must exist, checks both locations)
            prompt = PathResolver.resolve("config/prompts/system.jinja2", must_exist=True)

            # Absolute path (used as-is)
            abs_path = PathResolver.resolve("/tmp/output")
        """
        # Convert to Path and expand user home directory (~/)
        path = Path(path).expanduser()

        # Absolute paths: use as-is
        if path.is_absolute():
            if must_exist and not path.exists():
                raise FileNotFoundError(f"Path not found: {path}")
            return path

        # Relative paths: check working dir and package dir
        cwd_path = Path.cwd() / path
        package_path = PathResolver.get_project_root() / path

        # For creation, always prefer working directory (user's context)
        if create_if_missing:
            return cwd_path

        # For reading, check both locations (working dir first)
        if cwd_path.exists():
            return cwd_path
        if package_path.exists():
            return package_path

        # Not found in either location
        if must_exist:
            raise FileNotFoundError(
                f"Path not found in:\n"
                f"  - Working dir: {cwd_path}\n"
                f"  - Package dir: {package_path}"
            )

        # Default to working dir (user's context)
        return cwd_path
