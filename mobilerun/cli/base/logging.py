"""Track the CLI's installed log handler across command groups."""

from mobilerun.logs.handlers import CLILogHandler

_handler: CLILogHandler | None = None


def remember_cli_handler(handler: CLILogHandler) -> None:
    global _handler
    _handler = handler


def cli_handler() -> CLILogHandler:
    global _handler
    if _handler is None:
        _handler = CLILogHandler()
    return _handler
