"""Portal auto-setup helpers."""

import inspect
from typing import Any, Callable

from mobilerun import __version__


def portal_version_kwargs(setup_fn: Callable[..., Any]) -> dict[str, str]:
    """Key the Portal version map by mobilerun's version when core-local allows it."""
    if "version" in inspect.signature(setup_fn).parameters:
        return {"version": __version__}
    return {}
