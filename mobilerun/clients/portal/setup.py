"""Portal auto-setup helpers."""

import inspect
from typing import Any, Callable

from mobilerun import __version__


def portal_version_kwargs(
    setup_fn: Callable[..., Any], *, in_place: bool = False
) -> dict[str, Any]:
    """Key the Portal version map by mobilerun's version when core-local allows it.

    With in_place, also ask for an update that keeps Portal's data.
    """
    params = inspect.signature(setup_fn).parameters
    kwargs: dict[str, Any] = {}
    if "version" in params:
        kwargs["version"] = __version__
    if in_place and "uninstall" in params:
        kwargs["uninstall"] = False
    return kwargs
