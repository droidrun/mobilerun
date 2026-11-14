"""Utility modules for Mobilerun agents."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mobilerun.agent.utils.chat_utils import (
        filter_empty_messages,
        has_content,
        limit_history,
        to_chat_messages,
    )
    from mobilerun.agent.utils.prompt_resolver import PromptResolver
    from mobilerun.agent.utils.signatures import build_tool_registry
    from mobilerun.agent.utils.trajectory import Trajectory

_EXPORTS = {
    "filter_empty_messages": (
        "mobilerun.agent.utils.chat_utils",
        "filter_empty_messages",
    ),
    "has_content": ("mobilerun.agent.utils.chat_utils", "has_content"),
    "limit_history": ("mobilerun.agent.utils.chat_utils", "limit_history"),
    "to_chat_messages": ("mobilerun.agent.utils.chat_utils", "to_chat_messages"),
    "PromptResolver": ("mobilerun.agent.utils.prompt_resolver", "PromptResolver"),
    "build_tool_registry": ("mobilerun.agent.utils.signatures", "build_tool_registry"),
    "Trajectory": ("mobilerun.agent.utils.trajectory", "Trajectory"),
}

__all__ = [
    # Chat utilities
    "to_chat_messages",
    "has_content",
    "filter_empty_messages",
    "limit_history",
    # Prompt utilities
    "PromptResolver",
    # Tool utilities
    "build_tool_registry",
    # Trajectory
    "Trajectory",
]


def __getattr__(name):
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        value = getattr(import_module(module), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
