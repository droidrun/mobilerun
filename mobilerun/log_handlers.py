"""
Logging handlers for Droidrun.

Provides CLILogHandler (Rich colored terminal output) and TUILogHandler
(record buffer for deferred Textual rendering). Both support extra params:

    color:      logger.info("msg", extra={"color": "blue"})
    stream:     logger.info(token, extra={"stream": True})
    stream_end: logger.info("", extra={"stream_end": True})
"""

import logging

from rich.console import Console

COLORS = frozenset(
    {"blue", "cyan", "green", "red", "yellow", "magenta", "white", "dim"}
)


def configure_logging(debug: bool, handler: logging.Handler) -> None:
    """Replace all handlers on the ``droidrun`` logger."""
    logger = logging.getLogger("droidrun")
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False


class CLILogHandler(logging.Handler):
    """Rich-colored line output for CLI and SDK use."""

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.console = Console()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            color = getattr(record, "color", None)
            stream = getattr(record, "stream", False)
            stream_end = getattr(record, "stream_end", False)

            if stream:
                # style= applies color without markup parsing
                self.console.print(msg, end="", highlight=False, markup=False)
            elif stream_end:
                self.console.print("", highlight=False)
            elif color and color in COLORS:
                self.console.print(msg, style=color, highlight=False, markup=False)
            else:
                self.console.print(msg, highlight=False, markup=False)
        except Exception:
            self.handleError(record)


class TUILogHandler(logging.Handler):
    """Captures log records for TUI rendering.

    Args:
        on_record: Optional callback invoked for every record with the record dict.
    """

    def __init__(self, on_record=None, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.on_record = on_record
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            color = getattr(record, "color", None)
            stream = getattr(record, "stream", False)
            stream_end = getattr(record, "stream_end", False)

            rec = {
                "msg": msg,
                "color": color,
                "stream": stream,
                "stream_end": stream_end,
                "level": record.levelno,
            }
            self.records.append(rec)

            if self.on_record:
                self.on_record(rec)
        except Exception:
            self.handleError(record)
