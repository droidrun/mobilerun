"""Dropdown widget for slash command selection."""

from __future__ import annotations

from textual.widget import Widget
from textual.message import Message
from textual.app import RenderResult
from textual.reactive import reactive
from rich.text import Text

from droidrun.cli.tui.commands import Command


class CommandDropdown(Widget):
    can_focus = False

    class Selected(Message):
        def __init__(self, command_name: str) -> None:
            super().__init__()
            self.command_name = command_name

    highlighted: reactive[int] = reactive(0)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._commands: list[Command] = []

    def update_commands(self, commands: list[Command]) -> None:
        self._commands = commands
        self.highlighted = 0
        self.refresh()

    def move_highlight(self, direction: int) -> None:
        if not self._commands:
            return
        new_idx = self.highlighted + direction
        self.highlighted = max(0, min(new_idx, len(self._commands) - 1))

    def select_highlighted(self) -> None:
        if self._commands and 0 <= self.highlighted < len(self._commands):
            cmd = self._commands[self.highlighted]
            self.post_message(self.Selected(cmd.name))

    @property
    def has_commands(self) -> bool:
        return bool(self._commands)

    def render(self) -> RenderResult:
        if not self._commands:
            return Text()

        # Compute column width from longest command name
        name_width = max(len(c.name) for c in self._commands) + 2
        try:
            total_width = self.size.width - 4  # account for padding
        except Exception:
            total_width = 60

        output = Text()
        for i, cmd in enumerate(self._commands):
            if i > 0:
                output.append("\n")

            is_highlighted = i == self.highlighted
            bg = " on #2e2e4a" if is_highlighted else ""
            name_style = f"bold #CAD3F6{bg}"
            desc_style = f"#838BBC{bg}" if not is_highlighted else f"#CAD3F6{bg}"
            slash_style = f"#47475e{bg}" if not is_highlighted else f"#838BBC{bg}"

            line_content = f"/{cmd.name:<{name_width}}{cmd.description}"
            pad = max(0, total_width - len(line_content) - 1)

            output.append(" /", style=slash_style)
            output.append(f"{cmd.name:<{name_width}}", style=name_style)
            output.append(cmd.description, style=desc_style)
            if is_highlighted:
                output.append(" " * pad, style=f"{bg}")

        return output
