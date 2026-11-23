"""UI snapshot value types."""

from typing import TypedDict


class UIElement(TypedDict, total=False):
    """Element data exposed by UI state snapshots."""

    index: int
    text: str
    className: str
    resourceId: str
    bounds: str
    checkedState: str
    displayBounds: str
    tapBlockers: list[str]
    children: list["UIElement"]
