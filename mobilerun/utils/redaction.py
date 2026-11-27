"""Task-local redaction of sensitive values in logs."""

import json
import logging
import re
import traceback
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit
from weakref import WeakSet

from pydantic import JsonValue

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


def _variants(values: tuple[str, ...], parent: _Variants | None) -> _Variants | None:
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
    return _redact_variants(text, _active_secrets.get())


def _redact_variants(text: str, variants: _Variants | None) -> str:
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


class KnownCodes:
    """Device-local verification codes retained only in memory."""

    def __init__(self) -> None:
        self._variants: _Variants | None = None
        self.sensitive_values: set[str] = set()
        self.observed_secrets: set[str] = set()

    def remember(self, *codes: str) -> None:
        self._variants = _variants(codes, self._variants)
        if self._variants is not None:
            self._enable_log_masking()

    def remember_sensitive(self, values: Iterable[str]) -> None:
        self.sensitive_values.update(filter(None, values))
        if self.sensitive_values:
            self._enable_log_masking()

    def _enable_log_masking(self) -> None:
        _known_codes.add(self)
        _install_secret_factory()
        _install_extra_redaction()

    def redact(self, text: str) -> str:
        return _redact_variants(text, self._variants)

    def redact_sensitive_text(self, text: str, *, include_observed: bool = True) -> str:
        values = (
            self.sensitive_values | self.observed_secrets
            if include_observed
            else self.sensitive_values
        )
        for value in sorted(values, key=len, reverse=True):
            text = re.sub(rf"(?<!\w){re.escape(value)}(?!\w)", "***", text)
        return text


_known_codes: WeakSet[KnownCodes] = WeakSet()


def _redact_log(text: str) -> str:
    text = _redact_secrets(text)
    for codes in tuple(_known_codes):
        text = codes.redact(codes.redact_sensitive_text(text, include_observed=False))
    return text


def mask_step_args(
    args: dict[str, JsonValue], *, contains_code: Callable[[str], bool] | None = None
) -> dict[str, JsonValue]:
    """Mask active secrets and device-known codes throughout a JSON argument tree."""

    def mask_dict(values: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {
            (
                "***"
                if contains_code is not None and contains_code(key)
                else _redact_secrets(key)
            ): mask(value)
            for key, value in values.items()
        }

    def mask(value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            return mask_dict(value)
        if isinstance(value, list):
            return [mask(item) for item in value]
        if value is not None:
            text = str(value)
            if _redact_secrets(text) != text or (
                contains_code is not None and contains_code(text)
            ):
                return "***"
        return value

    return mask_dict(args)


def is_secure_ui_node(node: dict[str, JsonValue]) -> bool:
    """Recognize Android, iOS, and generic secure input markers."""
    for key, value in node.items():
        name = key.lower().replace("_", "").replace("-", "")
        text = str(value).lower() if isinstance(value, (str, int)) else ""
        if name in {
            "ispassword",
            "password",
            "secure",
            "issecure",
            "securetextentry",
            "issecuretextentry",
        } and text in {"true", "1", "yes"}:
            return True
        if name in {"type", "inputtype", "classname", "class"}:
            if "password" in text or "securetextfield" in text:
                return True
            if name == "inputtype":
                try:
                    if int(text, 0) & 0xFF in {0x12, 0x81, 0x91, 0xE1}:
                        return True
                except ValueError:
                    pass
    details = node.get("details")
    return isinstance(details, dict) and is_secure_ui_node(details)


def mask_secure_ui(
    ui: dict[str, JsonValue], *, retain_targeting: bool = False
) -> dict[str, JsonValue]:
    """Remove secure values and their copies; optionally keep live targets."""
    protected = False
    values: set[str] = set()
    targeting = (
        {"index", "bounds", "displayBounds", "className", "resourceId", "tapBlockers"}
        if retain_targeting
        else set()
    )

    def remove(value: JsonValue) -> JsonValue:
        nonlocal protected
        if isinstance(value, dict):
            if is_secure_ui_node(value):
                protected = True
                for name, item in value.items():
                    if name not in targeting:
                        collect(item)
                return {
                    **{name: item for name, item in value.items() if name in targeting},
                    "isPassword": True,
                    "text": "***",
                }
            return {key: remove(item) for key, item in value.items()}
        if isinstance(value, list):
            return [remove(item) for item in value]
        return value

    def collect(value: JsonValue) -> None:
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str) and value:
            values.add(value)

    elements = remove(ui.get("elements"))
    remove(ui.get("phone_state"))
    if not protected:
        return ui
    # Both summaries are independently rendered and may repeat a password.
    # Phone state can also carry a separately formatted focused-element copy.
    return {
        **ui,
        "elements": mask_step_args(
            {"elements": elements},
            contains_code=lambda text: any(value in text for value in values),
        ).get("elements", []),
        "phone_state": {},
        "formatted_text": "***",
        "focused_text": "***",
    }


def _install_secret_factory() -> None:
    previous = logging.getLogRecordFactory()
    if getattr(previous, "_mobilerun_secret_redaction", False):
        return

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        if _active_secrets.get() or _known_codes:
            record.msg = _redact_log(record.getMessage())
            record.args = ()
            if record.exc_info:
                record.exc_text = _redact_log(
                    "".join(traceback.format_exception(*record.exc_info))
                )
                record.exc_info = None
            elif record.exc_text:
                record.exc_text = _redact_log(record.exc_text)
            if record.stack_info:
                record.stack_info = _redact_log(record.stack_info)
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
        if extra and (_active_secrets.get() or _known_codes):
            for key in extra:
                value = record.__dict__[key]
                rendered = str(value)
                redacted = _redact_log(rendered)
                if rendered != redacted:
                    record.__dict__[key] = redacted
                elif not isinstance(value, str):
                    representation = repr(value)
                    redacted_repr = _redact_log(representation)
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
    token = _active_secrets.set(_variants(values, _active_secrets.get()))
    try:
        yield
    finally:
        _active_secrets.reset(token)


def mask_url(url: str) -> str:
    """Show an HTTP(S) destination without userinfo, query values or fragment.

    The path is shown on purpose: users want to recognise their own URLs in
    `mobilerun webhook list`. Services that put the secret in the path
    (webhook.site, Slack, Discord) are therefore shown in full; only use this
    for output the user explicitly asked to see, never for logs or errors.
    """
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
