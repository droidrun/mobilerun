"""Connection identifiers safe to include in action logs."""

import json
import logging
import re
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit


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
_active_secrets: ContextVar[tuple[str, ...]] = ContextVar(
    "mobilerun_input_secrets", default=()
)


def _redact_secrets(text: str) -> str:
    for secret in _active_secrets.get():
        if secret:
            variants = {
                secret,
                repr(secret)[1:-1],
                json.dumps(secret, ensure_ascii=True)[1:-1],
                json.dumps(secret, ensure_ascii=False)[1:-1],
                quote(secret, safe=""),
                quote_plus(secret, safe=""),
            }
            for variant in sorted(variants, key=len, reverse=True):
                text = text.replace(variant, "***")
    return text


def _install_secret_factory() -> None:
    previous = logging.getLogRecordFactory()
    if getattr(previous, "_mobilerun_secret_redaction", False):
        return

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        if _active_secrets.get():
            record.msg = _redact_secrets(record.getMessage())
            record.args = ()
            if record.exc_info:
                record.exc_text = _redact_secrets(
                    "".join(traceback.format_exception(*record.exc_info))
                )
                record.exc_info = None
            elif record.exc_text:
                record.exc_text = _redact_secrets(record.exc_text)
            if record.stack_info:
                record.stack_info = _redact_secrets(record.stack_info)
        return record

    setattr(factory, "_mobilerun_secret_redaction", True)  # noqa: B010
    logging.setLogRecordFactory(factory)


def _install_extra_redaction() -> None:
    previous = logging.Logger.makeRecord
    if getattr(previous, "_mobilerun_extra_redaction", False):
        return

    def make_record(
        self,
        name,
        level,
        fn,
        lno,
        msg,
        args,
        exc_info,
        func=None,
        extra=None,
        sinfo=None,
    ):
        record = previous(
            self, name, level, fn, lno, msg, args, exc_info, func, extra, sinfo
        )
        if extra and _active_secrets.get():
            for key in extra:
                value = record.__dict__[key]
                rendered = str(value)
                redacted = _redact_secrets(rendered)
                if rendered != redacted:
                    record.__dict__[key] = redacted
                elif not isinstance(value, str):
                    representation = repr(value)
                    redacted_repr = _redact_secrets(representation)
                    if representation != redacted_repr:
                        record.__dict__[key] = redacted_repr
        return record

    setattr(make_record, "_mobilerun_extra_redaction", True)  # noqa: B010
    logging.Logger.makeRecord = make_record


@contextmanager
def secret_scope(*values: str) -> Iterator[None]:
    """Redact values in this task; use asyncio.to_thread for blocking work.

    Bare run_in_executor does not propagate context variables. If it must be
    used, submit contextvars.copy_context().run with the callable instead.
    """
    _install_secret_factory()
    _install_extra_redaction()
    token = _active_secrets.set((*_active_secrets.get(), *values))
    try:
        yield
    finally:
        _active_secrets.reset(token)


_secret_input = secret_scope


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
