import logging
import sys

from mobilerun.logs.handlers import configure_logging


def log_progress_to_stderr() -> None:
    """Show INFO progress (e.g. a slow UI-state fetch) on stderr.

    Without this, Python's last-resort handler drops everything below
    WARNING, so a slow device looks like a silent hang. stdout stays
    reserved for the command's output.
    """
    if logging.getLogger("mobilerun").handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    configure_logging(debug=False, handler=handler)
