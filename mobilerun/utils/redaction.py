"""Task-local redaction of sensitive values in logs."""

import json
import logging
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from urllib.parse import quote, quote_plus

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

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
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
        self: logging.Logger,
        name: Any,
        level: Any,
        fn: Any,
        lno: Any,
        msg: Any,
        args: Any,
        exc_info: Any,
        func: Any = None,
        extra: Any = None,
        sinfo: Any = None,
    ) -> logging.LogRecord:
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
    logging.Logger.makeRecord = make_record  # type: ignore[method-assign]


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
