from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mobilerun.agent.droid.events import ResultEvent

_EXPORTS = {
    "ResultEvent": ("mobilerun.agent.droid.events", "ResultEvent"),
}

__all__ = ["ResultEvent"]


def __getattr__(name):
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        value = getattr(import_module(module), attribute)
        globals()[name] = value
        return value
    if name in {"droid", "utils"}:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
