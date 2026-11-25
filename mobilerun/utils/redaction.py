"""Task-local redaction of sensitive values in logs."""

import json
import logging
import re
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit

# A period immediately following a digit only joins another digit, not a new
# sentence. Possessive runs consume each span of spaces/digits without backtracking.
_NUMERIC_RUN = re.compile(r"\d++(?:(?: ++(?:[./-] *+)?|[/-] *+|\.)\d++)*")


@dataclass(frozen=True, repr=False)
class _Variants:
    literals: frozenset[str]
    digits: frozenset[str]


_active_secrets: ContextVar[_Variants | None] = ContextVar(
    "mobilerun_input_secrets", default=None
)


def _variants(values: tuple[str, ...]) -> _Variants | None:
    parent = _active_secrets.get()
    literals = set(parent.literals) if parent else set()
    numbers = set(parent.digits) if parent else set()
    for value in filter(None, values):
        forms = {
            value,
            repr(value)[1:-1],
            json.dumps(value)[1:-1],
            json.dumps(value, ensure_ascii=False)[1:-1],
        }
        try:
            forms.update((quote(value, safe=""), quote_plus(value, safe="")))
        except UnicodeEncodeError:
            pass  # Lone surrogates have no UTF-8 URL encoding.
        digits = "".join(c for c in value if c.isdecimal())
        if len(digits) >= 4 and all(c.isdecimal() or c in " ./-" for c in value):
            numbers.add(digits)
            if _NUMERIC_RUN.fullmatch(value):
                # Numeric literals use group boundaries too, e.g. 12/27 in a date.
                forms.discard(value)
        literals.update(forms)
    return (
        _Variants(frozenset(literals), frozenset(numbers))
        if literals or numbers
        else None
    )


def _numeric_spans(text: str, numbers: frozenset[str]) -> Iterator[tuple[int, int]]:
    lengths = sorted({len(value) for value in numbers})
    for match in _NUMERIC_RUN.finditer(text):
        run = match.group()
        if len(run) < lengths[0]:
            continue
        if run.isdecimal():
            if run in numbers:
                yield match.span()
            continue
        # Only whitespace between groups is a boundary. Spaces surrounding
        # punctuation belong to the separator and do not delimit a match.
        digits: list[str] = []
        starts = [(0, match.start())]
        ends = {}
        last_digit = -1
        for index, char in enumerate(run):
            if char.isdecimal():
                if run[last_digit + 1 : index].isspace():
                    ends[len(digits)] = match.start() + last_digit + 1
                    starts.append((len(digits), match.start() + index))
                digits.append(char)
                last_digit = index
        ends[len(digits)] = match.end()
        compact = "".join(digits)
        for offset, start in starts:
            for length in lengths:
                stop = offset + length
                if stop > len(compact):
                    break
                if stop in ends and compact[offset:stop] in numbers:
                    yield start, ends[stop]


def _redact_secrets(text: str) -> str:
    variants = _active_secrets.get()
    if not variants:
        return text
    spans = list(_numeric_spans(text, variants.digits)) if variants.digits else []
    for literal in variants.literals:
        offset = 0
        while (start := text.find(literal, offset)) >= 0:
            spans.append((start, start + len(literal)))
            offset = start + 1
    if not spans:
        return text
    parts: list[str] = []
    end = 0
    for start, stop in sorted(spans):
        if start >= end:
            parts.extend((text[end:start], "***"))
        end = max(end, stop)
    return "".join([*parts, text[end:]])


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
    token = _active_secrets.set(_variants(values))
    try:
        yield
    finally:
        _active_secrets.reset(token)


def mask_url(url: str) -> str:
    """Show an HTTP(S) destination without userinfo, query values or fragment."""
    try:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return "[redacted]"
    except ValueError:
        return "[redacted]"
    query = "&".join(
        item.partition("=")[0] + "=•••" if "=" in item else "•••"
        for item in parts.query.split("&")
        if item
    )
    return urlunsplit(
        (parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, query, "")
    )
