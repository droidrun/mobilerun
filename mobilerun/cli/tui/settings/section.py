"""Reusable widgets for settings tabs."""

from __future__ import annotations

from textual.containers import VerticalGroup
from textual.reactive import reactive
from textual.widgets import Static


class Section(VerticalGroup):
    """A visually grouped section with a border title."""

    def __init__(
        self,
        title: str,
        hint: str = "",
        *children,
        **kwargs,
    ) -> None:
        super().__init__(*children, **kwargs)
        self.border_title = title
        if hint:
            self.border_subtitle = hint


class BoolToggle(Static):
    """A clickable true/false toggle."""

    value: reactive[bool] = reactive(False)

    def __init__(self, value: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value

    def render(self) -> str:
        return "● true" if self.value else "○ false"

    def on_click(self) -> None:
        self.value = not self.value

    def watch_value(self, value: bool) -> None:
        if value:
            self.add_class("-on")
        else:
            self.remove_class("-on")

    def on_mount(self) -> None:
        if self.value:
            self.add_class("-on")
