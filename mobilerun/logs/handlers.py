"""
Logging handlers for Mobilerun.

Provides CLILogHandler (Rich colored terminal output) and TUILogHandler
(record buffer for deferred Textual rendering). Both support extra params:

    color:      logger.info("msg", extra={"color": "blue"})
    stream:     logger.info(token, extra={"stream": True})
    stream_end: logger.info("", extra={"stream_end": True})
"""

import logging
from collections.abc import Callable
from typing import Literal, Protocol, TypeAlias, cast

from rich.console import Console

COLORS = frozenset(
    {"blue", "cyan", "green", "red", "yellow", "magenta", "white", "dim"}
)
LogRecordData: TypeAlias = dict[str, str | bool | int | None]
LogLevel: TypeAlias = Literal["quiet", "normal", "verbose", "debug"]
VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")
LOG_LEVELS: dict[LogLevel, int] = {
    "quiet": logging.WARNING,
    "normal": logging.INFO,
    "verbose": VERBOSE,
    "debug": logging.DEBUG,
}


class LoggingSettings(Protocol):
    level: LogLevel
    debug: bool


def resolve_log_level(
    cli: LogLevel | None, env: str | None, config: LoggingSettings
) -> LogLevel:
    """Resolve CLI, environment, then config, including legacy debug settings."""
    value = (
        cli
        if cli is not None
        else env if env is not None else ("debug" if config.debug else config.level)
    )
    if value not in LOG_LEVELS:
        raise ValueError(
            f"Invalid log level: {value!r}; expected quiet, normal, verbose, or debug"
        )
    return cast(LogLevel, value)


def apply_log_level(
    config: LoggingSettings, cli: LogLevel | None = None, debug: bool | None = None
) -> LogLevel:
    """Apply one resolved verbosity to the configuration and shared logger."""
    from mobilerun.config.settings import get_settings

    explicit = (
        cli
        if cli is not None
        else ("debug" if debug is True else "normal" if debug is False else None)
    )
    env = get_settings().MOBILERUN_LOG_LEVEL if explicit is None else None
    level = resolve_log_level(explicit, env, config)
    config.level = level
    config.debug = level == "debug"
    logging.getLogger("mobilerun").setLevel(LOG_LEVELS[level])
    return level


def log_verbose(
    logger: logging.Logger, message: str, **extra: str | bool | int | None
) -> None:
    """Log diagnostic output without showing it at the default level."""
    logger.log(VERBOSE, message, extra=extra)


def ensure_agent_logging() -> None:
    """Provide SDK output on first agent use, respecting application logging."""
    logger = logging.getLogger("mobilerun")
    if logger.handlers or logging.getLogger().handlers:
        return
    logger.addHandler(CLILogHandler())
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    logger.propagate = False


def configure_logging(
    debug: bool, handler: logging.Handler, level: LogLevel | None = None
) -> None:
    """Replace all handlers on the ``mobilerun`` logger."""
    logger = logging.getLogger("mobilerun")
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(
        LOG_LEVELS[level]
        if level is not None
        else logging.DEBUG if debug else logging.INFO
    )
    logger.propagate = False


class CLILogHandler(logging.Handler):
    """Rich-colored line output for CLI and SDK use."""

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.console = Console(stderr=True)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # These optional fields are supplied through logging's dynamic extras.
            color = getattr(record, "color", None)
            result_status = getattr(record, "result_status", None)
            symbol = (
                {"success": "✓", "failure": "✗"}.get(result_status)
                if isinstance(result_status, str)
                else None
            )
            if symbol:
                try:
                    symbol.encode(self.console.encoding)
                except (UnicodeEncodeError, LookupError):
                    pass
                else:
                    msg = f"{symbol} {msg}"
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

    def __init__(
        self,
        on_record: Callable[[LogRecordData], None] | None = None,
        level: int = logging.NOTSET,
    ) -> None:
        super().__init__(level)
        self.on_record = on_record
        self.records: list[LogRecordData] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # These optional fields are supplied through logging's dynamic extras.
            color = getattr(record, "color", None)
            stream = getattr(record, "stream", False)
            stream_end = getattr(record, "stream_end", False)

            rec: LogRecordData = {
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
