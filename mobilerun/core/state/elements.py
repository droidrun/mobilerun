"""Typed representation of elements emitted by the UI tree formatters."""

from typing import TypedDict

from mobilerun.utils.geometry import Bounds


def _parse_bounds(value: str) -> Bounds:
    left, top, right, bottom = map(int, value.split(","))
    return left, top, right, bottom


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
