"""Redaction of typed secrets from everything the agent observes.

Three pieces share one :class:`SecretRedactor` per agent:

* :class:`SecretSafeDriver` wraps the (already Stealth/Recording wrapped)
  driver. UI trees pass through the redactor, screenshots turn black once an
  approval-gated secret was typed, and secrets are typed through
  :meth:`SecretSafeDriver.input_secret`, which keeps the plaintext out of the
  recording log.
* :class:`SecretSafeStateProvider` redacts the ``UIState`` returned by any
  state provider (built-in or host-injected) at the engine boundary, without
  touching the host-owned provider.
* :class:`SecretLogFilter` masks registered values in the rendered output of
  the engine loggers' handlers, including exception tracebacks.

Coverage is the built-in workflows and cooperating providers. A host that
reads the raw driver itself, or an external agent with raw ADB access, is
outside this boundary.
"""

from __future__ import annotations

import contextvars
import copy
import io
import logging
import re
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, List, Literal, Optional, Pattern

logger = logging.getLogger("mobilerun")

PLACEHOLDER = "[secret:{}]"
_MIN_GLOBAL_DIGITS = 8

# Geometry and identity fields: never carry secrets, and rewriting them would
# break tap targets.
_SKIP_KEYS = frozenset(
    {"bounds", "displayBounds", "boundsInScreen", "tapBlockers", "index", "windowId"}
)
# Keys that make a node "focused or editable" when truthy.
_INPUT_FLAG_KEYS = ("focused", "isFocused", "isEditable", "editable")
_INPUT_CLASS_MARKERS = ("EditText", "TextField", "SearchField")


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    secret_id: str
    pattern: Pattern[str]
    # Digit-only values of at least _MIN_GLOBAL_DIGITS digits are masked
    # everywhere, whatever the configured policy.
    everywhere: bool
    input_only: bool

    def applies(self, in_input: bool, structured: bool) -> bool:
        if self.everywhere or not self.input_only:
            return True
        # input_only only narrows where the tree says what is editable.
        return in_input or not structured


def _edge(char: str, *, leading: bool) -> str:
    look = "(?<!{})" if leading else "(?!{})"
    if char.isdigit():
        return look.format(r"\d")
    if char.isalnum() or char == "_":
        return look.format(r"\w")
    return ""


def _compile(value: str) -> tuple[Pattern[str], bool]:
    """Return (pattern, digits_only_long) for a secret value."""
    stripped = value.strip()
    digits = re.sub(r"[\s\-]", "", stripped)
    numeric_like = (
        bool(digits) and digits.isdigit() and bool(re.fullmatch(r"[\d\s\-]+", stripped))
    )
    if numeric_like:
        # Any grouping of the same digits: spaces or hyphens between any two.
        body = r"[\s\-]*".join(re.escape(d) for d in digits)
        pattern = rf"(?<!\d){body}(?!\d)"
        return re.compile(pattern), len(digits) >= _MIN_GLOBAL_DIGITS

    tokens = re.findall(r"\s+|\d+|[^\s\d]+", stripped)
    body = ""
    for i, token in enumerate(tokens):
        if token.isspace():
            body += r"\s+"
        elif (
            0 < i < len(tokens) - 1
            and re.fullmatch(r"[/\-.:]+", token)
            and tokens[i - 1].isdigit()
            and tokens[i + 1].isdigit()
        ):
            # Dates such as "MM / YY" may be shown with extra blanks.
            body += r"\s*" + re.escape(token) + r"\s*"
        else:
            body += re.escape(token)
    pattern = (
        _edge(stripped[0], leading=True) + body + _edge(stripped[-1], leading=False)
    )
    return re.compile(pattern, re.IGNORECASE), False


def _is_input_node(node: dict) -> bool:
    if any(
        node.get(key) is True or node.get(key) == "true" for key in _INPUT_FLAG_KEYS
    ):
        return True
    class_name = node.get("className") or node.get("class") or ""
    return isinstance(class_name, str) and any(
        marker in class_name for marker in _INPUT_CLASS_MARKERS
    )


def _tree_is_structured(node: Any) -> bool:
    """True when the tree says per node whether it is focused or editable."""
    if isinstance(node, list):
        return any(_tree_is_structured(child) for child in node)
    if not isinstance(node, dict):
        return False
    if any(key in node for key in _INPUT_FLAG_KEYS):
        return True
    return _tree_is_structured(node.get("children") or [])


class SecretRedactor:
    """Per-run registry of typed secret values and the masking policy."""

    def __init__(self) -> None:
        self._entries: List[_Entry] = []
        self._lock = threading.Lock()
        self.black_screenshots = False
        _ACTIVE.add(self)

    # -- registry ----------------------------------------------------------

    def register(
        self,
        secret_id: str,
        value: str,
        *,
        redact: Literal["global", "input_only"] = "global",
        protected: bool = False,
    ) -> None:
        """Register a value that is about to be typed (before typing it)."""
        if not value or not value.strip():
            return
        pattern, long_digits = _compile(value)
        entry = _Entry(
            secret_id=secret_id,
            pattern=pattern,
            everywhere=long_digits,
            input_only=redact == "input_only",
        )
        with self._lock:
            self._entries.append(entry)
            # Longest pattern first so overlapping values mask the longer one.
            self._entries.sort(key=lambda e: len(e.pattern.pattern), reverse=True)
            if protected:
                self.black_screenshots = True

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self.black_screenshots = False

    @property
    def active(self) -> bool:
        return bool(self._entries)

    # -- masking -----------------------------------------------------------

    def redact_text(
        self, text: str, *, in_input: bool = False, structured: bool = False
    ) -> str:
        entries = self._entries
        if not entries or not text:
            return text
        for entry in list(entries):
            if entry.applies(in_input, structured):
                text = entry.pattern.sub(PLACEHOLDER.format(entry.secret_id), text)
        return text

    def redact_data(
        self, data: Any, *, in_input: bool = False, structured: bool = False
    ) -> Any:
        """Return a copy of nested data with every string value masked."""
        if not self._entries:
            return data
        if isinstance(data, str):
            return self.redact_text(data, in_input=in_input, structured=structured)
        if isinstance(data, list):
            return [
                self.redact_data(item, in_input=in_input, structured=structured)
                for item in data
            ]
        if isinstance(data, tuple):
            return tuple(
                self.redact_data(item, in_input=in_input, structured=structured)
                for item in data
            )
        if isinstance(data, dict):
            node_input = in_input or (structured and _is_input_node(data))
            return {
                key: (
                    value
                    if key in _SKIP_KEYS
                    else self.redact_data(
                        value,
                        in_input=(
                            True
                            if key == "focusedElement"
                            else (node_input if key != "children" else in_input)
                        ),
                        structured=structured,
                    )
                )
                for key, value in data.items()
            }
        return data

    def redact_ui_tree(self, data: Any) -> Any:
        """Redact the payload of ``driver.get_ui_tree()``."""
        if not self._entries or not isinstance(data, dict):
            return data
        structured = _tree_is_structured(data.get("a11y_tree"))
        return self.redact_data(data, structured=structured)

    def redact_ui_state(self, state: Any, *, trusted: bool = False) -> Any:
        """Redact a ``UIState`` (copy; the input is not modified).

        ``trusted`` is for states built from this redactor's own driver, whose
        tree was already masked with the structured ``input_only`` policy; only
        the context-free values are masked again. For anything else the state
        carries no editable/focused information, so ``input_only`` values are
        masked everywhere (more masking, never less).
        """
        if not self._entries or state is None:
            return state
        structured = trusted
        masked = copy.copy(state)
        for attr in ("elements", "phone_state"):
            if hasattr(state, attr):
                setattr(
                    masked,
                    attr,
                    self.redact_data(getattr(state, attr), structured=structured),
                )
        for attr in ("formatted_text", "focused_text"):
            value = getattr(state, attr, None)
            if isinstance(value, str):
                setattr(
                    masked,
                    attr,
                    self.redact_text(
                        value, in_input=attr == "focused_text", structured=structured
                    ),
                )
        return masked


# Redactors that exist right now. Used by the log filter when a record is
# emitted outside any run's context.
_ACTIVE: "weakref.WeakSet[SecretRedactor]" = weakref.WeakSet()
_CURRENT: contextvars.ContextVar[Optional[SecretRedactor]] = contextvars.ContextVar(
    "mobilerun_secret_redactor", default=None
)


@contextmanager
def redaction_scope(redactor: SecretRedactor) -> Iterator[None]:
    """Bind ``redactor`` to the current context; tasks started inside inherit it."""
    token = _CURRENT.set(redactor)
    try:
        yield
    finally:
        _CURRENT.reset(token)


# ---------------------------------------------------------------------------
# Screenshots
# ---------------------------------------------------------------------------

_BLACK_1PX_PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"


def black_like(image: Any) -> Any:
    """Return a fully black PNG with the size of ``image`` (fail closed)."""
    if not isinstance(image, (bytes, bytearray)):
        return image
    try:
        from PIL import Image

        with Image.open(io.BytesIO(bytes(image))) as source:
            size = source.size
        out = io.BytesIO()
        Image.new("RGB", size, (0, 0, 0)).save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return _BLACK_1PX_PNG


# ---------------------------------------------------------------------------
# Driver wrapper
# ---------------------------------------------------------------------------


class SecretSafeDriver:
    """Outermost driver wrapper: redacts observations, protects secret input.

    Wrap order is ``raw -> Stealth -> Recording -> SecretSafe``. Recording and
    Stealth only touch mutating actions, so observations returned through them
    are still masked here, and secrets typed through :meth:`input_secret` are
    removed from the recording log right after they were typed.
    """

    def __init__(self, inner: Any, redactor: SecretRedactor) -> None:
        self.inner = inner
        self.redactor = redactor

    @property
    def platform(self) -> str:
        return self.inner.platform

    @property
    def supported(self) -> set[str]:
        return self.inner.supported

    @property
    def supported_buttons(self) -> set[str]:
        return self.inner.supported_buttons

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def log(self) -> list:
        """Redacted view of the recording log, if the chain has one."""
        raw = self.inner.log
        if not isinstance(raw, list):
            raise AttributeError("log")
        return [self.redactor.redact_data(dict(entry)) for entry in raw]

    # -- observations ------------------------------------------------------

    async def get_ui_tree(self, *args: Any, **kwargs: Any) -> Any:
        return self.redactor.redact_ui_tree(
            await self.inner.get_ui_tree(*args, **kwargs)
        )

    async def get_ui_tree_unredacted(self) -> Any:
        """Raw tree for engine-internal target checks. Never expose or log it."""
        return await self.inner.get_ui_tree()

    async def screenshot(self, *args: Any, **kwargs: Any) -> Any:
        image = await self.inner.screenshot(*args, **kwargs)
        if self.redactor.black_screenshots:
            return black_like(image)
        return image

    # -- input -------------------------------------------------------------

    async def input_secret(
        self,
        secret_id: str,
        value: str,
        *,
        redact: Literal["global", "input_only"] = "global",
        protected: bool = False,
        clear: bool = False,
    ) -> bool:
        """Type a secret. The value is registered for masking before typing."""
        self.redactor.register(secret_id, value, redact=redact, protected=protected)
        try:
            return await self.inner.input_text(value, clear)
        finally:
            self._scrub_recording_log(secret_id, value, protected)

    def _scrub_recording_log(self, secret_id: str, value: str, protected: bool) -> None:
        """Replace plaintext ``input_text`` entries of ``value`` in inner logs."""
        node = self.inner
        seen = 0
        while node is not None and seen < 8:
            seen += 1
            log = node.__dict__.get("log") if hasattr(node, "__dict__") else None
            if isinstance(log, list):
                for i, entry in enumerate(log):
                    if (
                        isinstance(entry, dict)
                        and entry.get("action_type") == "input_text"
                        and isinstance(entry.get("text"), str)
                        and value in entry["text"]
                    ):
                        replacement = {
                            "action_type": "type_secret",
                            "secret_id": secret_id,
                            "clear": bool(entry.get("clear", False)),
                        }
                        if protected:
                            replacement["protected"] = True
                        log[i] = replacement
            node = node.__dict__.get("inner") if hasattr(node, "__dict__") else None


# ---------------------------------------------------------------------------
# State provider proxy
# ---------------------------------------------------------------------------


class SecretSafeStateProvider:
    """Redacts the ``UIState`` of any state provider at the engine boundary.

    The wrapped provider is left untouched, so a host-owned provider that is
    shared between agents never changes behaviour. Every other attribute is
    forwarded.
    """

    def __init__(
        self, inner: Any, redactor: SecretRedactor, *, trusted: bool = False
    ) -> None:
        self.__dict__["inner"] = inner
        self.__dict__["redactor"] = redactor
        self.__dict__["trusted"] = trusted

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["inner"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.__dict__["inner"], name, value)

    async def get_state(self, *args: Any, **kwargs: Any) -> Any:
        state = await self.inner.get_state(*args, **kwargs)
        return self.redactor.redact_ui_state(state, trusted=self.trusted)

    async def get_final_state(self, *args: Any, **kwargs: Any) -> Any:
        state = await self.inner.get_final_state(*args, **kwargs)
        return self.redactor.redact_ui_state(state, trusted=self.trusted)


# ---------------------------------------------------------------------------
# Log filter
# ---------------------------------------------------------------------------


class SecretLogFilter(logging.Filter):
    """Masks registered values in the rendered output of engine log handlers.

    Installed on handlers (not loggers) so records from child loggers are
    covered too. Handles the formatted message and the traceback text.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        redactors = self._redactors()
        if not redactors:
            return True
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        masked = self._mask(message, redactors)
        if masked != message:
            record.msg = masked
            record.args = ()
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self._mask(record.exc_text, redactors)
        if record.stack_info:
            record.stack_info = self._mask(record.stack_info, redactors)
        return True

    @staticmethod
    def _redactors() -> List[SecretRedactor]:
        current = _CURRENT.get()
        candidates = [current] if current is not None else list(_ACTIVE)
        return [r for r in candidates if r.active]

    @staticmethod
    def _mask(text: str, redactors: List[SecretRedactor]) -> str:
        for redactor in redactors:
            # Rendered text has no tree structure: mask input_only values too.
            text = redactor.redact_text(text, structured=False)
        return text


_LOG_FILTER = SecretLogFilter()


def install_log_redaction(root: str = "mobilerun") -> None:
    """Attach the filter to every handler of the engine logger tree (idempotent)."""
    names = [root] + [
        name
        for name in list(logging.root.manager.loggerDict)
        if name.startswith(root + ".")
    ]
    for name in names:
        for handler in logging.getLogger(name).handlers:
            if _LOG_FILTER not in handler.filters:
                handler.addFilter(_LOG_FILTER)
