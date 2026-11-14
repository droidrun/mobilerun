"""Mobilerun Agent Module.

This module provides a ReAct agent for automating Android devices using reasoning and acting.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mobilerun.agent.droid.droid_agent import MobileAgent
    from mobilerun.agent.droid.state import MobileAgentState

_EXPORTS = {
    "MobileAgent": ("mobilerun.agent.droid.droid_agent", "MobileAgent"),
    "MobileAgentState": ("mobilerun.agent.droid.state", "MobileAgentState"),
}

__all__ = ["MobileAgent", "MobileAgentState", "DroidAgent", "DroidAgentState"]


def __getattr__(name):
    aliases = {"DroidAgent": "MobileAgent", "DroidAgentState": "MobileAgentState"}
    if name in aliases:
        import warnings

        warnings.warn(
            f"{name} has been renamed. Update your imports.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(import_module(__name__), aliases[name])
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        value = getattr(import_module(module), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
