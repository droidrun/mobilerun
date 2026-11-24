"""Selected optional runtime, loaded only when its exports are requested."""

from importlib import import_module

from .settings import RUNTIME_MODULE

try:
    _runtime = import_module(RUNTIME_MODULE)
except ModuleNotFoundError as exc:
    if exc.name != RUNTIME_MODULE:
        raise
    raise ImportError(
        "Sandbox support requires the optional dependency; "
        "install it with pip install 'mobilerun[sandbox]'"
    ) from exc


def __getattr__(name: str):
    """Forward selected runtime exports without copying their identity."""
    if name in _runtime.__all__:
        return getattr(_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
