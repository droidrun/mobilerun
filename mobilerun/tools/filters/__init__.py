"""Filter exports."""

from .base import TreeFilter
from .concise_filter import ConciseFilter
from .detailed_filter import DetailedFilter


def get_filter(name: str, **kwargs) -> TreeFilter:
    """Factory to get filter by name."""
    filters = {
        "concise": ConciseFilter,
        "detailed": DetailedFilter,
    }

    filter_class = filters.get(name.lower())
    if filter_class is None:
        raise ValueError(f"Unknown filter: {name}. Available: {list(filters.keys())}")

    return filter_class(**kwargs)


__all__ = ["TreeFilter", "ConciseFilter", "DetailedFilter", "get_filter"]
