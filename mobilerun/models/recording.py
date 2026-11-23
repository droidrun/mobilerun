"""Recorded device action values."""

from typing import Any, NotRequired, TypedDict


class RecordedAction(TypedDict):
    action_type: str
    x: NotRequired[int]
    y: NotRequired[int]
    start_x: NotRequired[int]
    start_y: NotRequired[int]
    end_x: NotRequired[int]
    end_y: NotRequired[int]
    duration_ms: NotRequired[float]
    text: NotRequired[str]
    clear: NotRequired[bool]
    button: NotRequired[str]
    package: NotRequired[str | None]
    activity: NotRequired[str | None]
    duration: NotRequired[float]
    field: NotRequired[str]
    name: NotRequired[str | None]
    arguments: NotRequired[dict[str, Any]]
    url: NotRequired[str]
    key_code: NotRequired[int]
