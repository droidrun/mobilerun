"""Connection identifiers safe to include in action logs."""

import logging
import re
from urllib.parse import urlsplit, urlunsplit


def _sanitize_identifier(identifier: str | None) -> str | None:
    """Keep ordinary device IDs; remove credentials and URL metadata."""
    if identifier is None or "://" not in identifier:
        return identifier
    try:
        parts = urlsplit(identifier)
        return urlunsplit(
            (parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", "")
        )
    except ValueError:
        # Malformed URLs must not fall back to logging the original secret.
        return "invalid-url"


_URL = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"'<>]+")


def _sanitize_urls(text: str) -> str:
    return _URL.sub(lambda match: _sanitize_identifier(match.group()) or "", text)


class _URLRedactionFilter(logging.Filter):
    _mobilerun_url_redaction = True

    def filter(self, record: logging.LogRecord) -> bool:
        # Format first so URLs split between message and arguments are covered too.
        record.msg = _sanitize_urls(record.getMessage())
        record.args = ()
        return True


_backend_logger = logging.getLogger("mobilerun_core_local")
if not any(
    getattr(item, "_mobilerun_url_redaction", False) for item in _backend_logger.filters
):
    _backend_logger.addFilter(_URLRedactionFilter())
