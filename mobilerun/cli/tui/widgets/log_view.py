"""Styled log view built on RichLog."""

from __future__ import annotations

from collections import deque

from textual.widgets import RichLog
from rich.text import Text

MAX_LOG_LINES = 10_000


class LogView(RichLog):
    """Colored log view with plain-text copy support.

    Uses RichLog for Rich-styled output (colored lines).
    Maintains a parallel plain-text buffer for clipboard copy.
    """

    ALLOW_SELECT = False

    def __init__(self, **kwargs) -> None:
        super().__init__(
            wrap=True,
            markup=False,
            auto_scroll=True,
            max_lines=MAX_LOG_LINES,
            **kwargs,
        )
        self._plain_lines: deque[str] = deque(maxlen=MAX_LOG_LINES)

    def append(self, line: str, style: str | None = None) -> None:
        """Append a line of text with optional color style."""
        text = Text(line)
        if style:
            text.stylize(style)
        self.write(text)
        self._plain_lines.append(line)

    def clear_log(self) -> None:
        """Clear all log content."""
        self.clear()
        self._plain_lines.clear()

    def get_plain_text(self) -> str:
        """Return all log content as plain text for clipboard."""
        return "\n".join(self._plain_lines)
