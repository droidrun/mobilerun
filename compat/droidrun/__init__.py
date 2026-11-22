"""Compatibility shim: droidrun -> mobilerun.

Uses a PEP 451 meta-path finder (find_spec) to lazily alias droidrun.*
imports to mobilerun.* on demand. Compatible with Python 3.11-3.13+.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings

warnings.warn(
    "The 'droidrun' package has been renamed to 'mobilerun'. "
    "Please update your imports. This compatibility shim will be "
    "removed in a future release.",
    FutureWarning,
    stacklevel=2,
)

# DO NOT import mobilerun here — that would eagerly trigger the full
# package initialization (log handlers, agent/config/macro/tool imports).
# Instead, use lazy __getattr__ for top-level symbols and a PEP 451
# meta-path finder for submodule aliasing.

# Names that are physical files within the compat package — they must NOT be
# intercepted by the alias finder. Everything else under ``droidrun.*`` maps
# to ``mobilerun.*``.
_COMPAT_OWN_NAMES = frozenset(
    {
        "droidrun.__main__",
        "droidrun.cli_shim",
        "droidrun.macro",
        "droidrun.macro.__main__",
    }
)


class _DroidrunAliasLoader(importlib.abc.Loader):
    """PEP 451 loader: returns the real mobilerun.* module object."""

    def __init__(self, real_name):
        self._real_name = real_name

    def create_module(self, spec):
        # Import the real module first so sys.modules[real_name] is populated.
        real_mod = importlib.import_module(self._real_name)
        self._real_spec = real_mod.__spec__
        # Eagerly alias under the droidrun.* name so subsequent recursive
        # imports (during create_module/exec_module) find the same object.
        sys.modules[spec.name] = real_mod
        return real_mod

    def exec_module(self, module):
        # The real module is already fully initialized; don't re-execute.
        # But re-assert the alias in case another finder overwrote it.
        sys.modules[module.__name__] = sys.modules.get(module.__name__, module)
        module.__spec__ = self._real_spec


class _DroidrunAliasFinder(importlib.abc.MetaPathFinder):
    """PEP 451 finder: lazily alias droidrun.* -> mobilerun.*.

    Uses find_spec (not find_module) — required for Python 3.12+/3.13.
    Physical compat files (__main__.py, cli_shim.py, macro/) are excluded
    so they load from the compat tree as normal.

    All other ``droidrun.X.Y...`` names are remapped to ``mobilerun.X.Y...``
    and the SAME module object is returned for both — this preserves class
    identity so ``droidrun.X.Y.Cls is mobilerun.X.Y.Cls`` holds.
    """

    _active = False  # re-entrancy guard during importlib.util.find_spec

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith("droidrun."):
            return None
        if fullname in sys.modules:
            return None
        if self._active:
            return None
        # Don't intercept the compat package's own physical files.
        if fullname in _COMPAT_OWN_NAMES:
            return None

        # Map to mobilerun.* and verify it exists.
        from mobilerun._legacy.aliases import legacy_module_name

        relative = fullname[len("droidrun.") :]
        new_name = legacy_module_name(relative) or "mobilerun." + relative
        self._active = True
        try:
            real_spec = importlib.util.find_spec(new_name)
        except (ImportError, ValueError):
            real_spec = None
        finally:
            self._active = False

        if real_spec is None:
            return None

        warnings.warn(
            f"{fullname} is deprecated; import {new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Build alias spec with correct package metadata. Module identity is
        # preserved because our finder is inserted at the front of
        # ``sys.meta_path``, so it intercepts every ``droidrun.X.Y.Z`` lookup
        # before Python's PathFinder can discover the .py file via the
        # parent's __path__ and create a duplicate module.
        is_package = real_spec.submodule_search_locations is not None
        alias_spec = importlib.machinery.ModuleSpec(
            fullname,
            _DroidrunAliasLoader(new_name),
            origin=real_spec.origin,
            is_package=is_package,
        )
        if is_package:
            alias_spec.submodule_search_locations = list(
                real_spec.submodule_search_locations
            )
        return alias_spec


# Insert at the FRONT of sys.meta_path so we intercept droidrun.* lookups
# before Python's PathFinder does — otherwise PathFinder would resolve
# ``droidrun.X.Y.Z`` via parent.__path__ (which points at the mobilerun tree)
# and load it as a duplicate module separate from ``mobilerun.X.Y.Z``.
sys.meta_path.insert(0, _DroidrunAliasFinder())


# Lazy top-level symbol forwarding — no `import mobilerun` at module scope.
# `import droidrun` alone does NOT trigger mobilerun's full init.
# Only `from droidrun import DroidAgent` (or similar) triggers it.
def __getattr__(name):
    """Forward any attribute access to mobilerun (loaded lazily)."""
    if name in {"MobileConfig", "DroidConfig"}:
        from mobilerun.agents.config.user_config import MobileConfig

        return MobileConfig
    import mobilerun as _real

    return getattr(_real, name)
