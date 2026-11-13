"""Redaction of registered secrets from everything the agent observes.

Three pieces share one :class:`SecretRedactor` per agent:

* :class:`SecretSafeDriver` wraps the (already Stealth/Recording wrapped)
  driver. UI trees pass through the redactor, screenshots turn black when
  start masking is requested or an approval-gated secret is typed. Input uses
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
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Literal, Optional, Pattern, Tuple

logger = logging.getLogger("mobilerun")

PLACEHOLDER = "[secret:{}]"
_MIN_GLOBAL_DIGITS = 8
_MIN_SEGMENTED_CHARS = 4
_CARD_NUMBER_RE = re.compile(r"[0-9]+(?:[\s\-]+[0-9]+)*")
# Subtrees that never hold screen text: not scanned for split values.
_NON_TEXT_KEYS = frozenset({"phone_state", "device_context", "focusedElement"})

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
    # Built from the secret value: keep out of repr/logs.
    pattern: Pattern[str] = field(repr=False)
    # Digit-only values of at least _MIN_GLOBAL_DIGITS digits are masked
    # everywhere, whatever the configured policy.
    everywhere: bool
    input_only: bool
    # Long enough to look for it spread over several adjacent text nodes.
    segmentable: bool
    # Characters of the value (plus separators) and its length without blanks:
    # a text node can only be part of a split value if it uses these characters.
    chars: frozenset = field(repr=False)
    max_compact: int = field(repr=False)
    # Prefix entry for partial input; dropped again when typing succeeded.
    partial: bool = False

    def applies(self, in_input: bool, structured: bool) -> bool:
        if self.everywhere or not self.input_only:
            return True
        # input_only only narrows where the tree says what is editable.
        return in_input or not structured

    def accepts(self, text: str) -> bool:
        """May ``text`` be one segment of this value?"""
        return all(c in self.chars or c.isspace() for c in text.lower())

    def key(self) -> tuple:
        return (
            self.secret_id,
            self.pattern.pattern,
            self.everywhere,
            self.input_only,
            self.partial,
        )


def _edge(char: str, *, leading: bool) -> str:
    look = "(?<!{})" if leading else "(?!{})"
    if char.isdigit():
        return look.format(r"\d")
    if char.isalnum() or char == "_":
        return look.format(r"\w")
    return ""


def _digits_of(value: str) -> Optional[str]:
    """The digits of a digit-only value (blanks and hyphens allowed), else None."""
    stripped = value.strip()
    digits = re.sub(r"[\s\-]", "", stripped)
    if digits and digits.isdigit() and re.fullmatch(r"[\d\s\-]+", stripped):
        return digits
    return None


def _luhn_valid(value: str) -> bool:
    digits = _digits_of(value)
    if digits is None:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _card_spans(text: str) -> Iterator[tuple[int, int]]:
    """Merged Luhn-valid, group-aligned windows of 13–19 digits.

    A compact overlong ID is one indivisible group. Separators use the same
    whitespace/hyphen rule as declared numeric values, including Unicode blanks.
    """
    pending = None
    for match in _CARD_NUMBER_RE.finditer(text):
        groups = []
        offset = match.start()
        for digits in re.split(r"[\s\-]+", match.group()):
            start = text.index(digits, offset)
            offset = start + len(digits)
            groups.append((start, offset, digits))
        for i, (start, _, _) in enumerate(groups):
            value = ""
            for _, end, digits in groups[i : i + 19]:
                if len(value) + len(digits) > 19:
                    break
                value += digits
                if len(value) < 13 or not _luhn_valid(value):
                    continue
                if pending is not None and start > pending[1]:
                    yield pending
                    pending = None
                pending = (
                    start if pending is None else pending[0],
                    end if pending is None else max(end, pending[1]),
                )
    if pending is not None:
        yield pending


def _compile(value: str) -> tuple[Pattern[str], bool]:
    """Return (pattern, digits_only_long) for a secret value."""
    stripped = value.strip()
    digits = _digits_of(value)
    numeric_like = digits is not None
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


# Other texts a node may repeat its own text in, piece by piece.
_ECHO_KEYS = ("contentDescription", "hint", "stateDescription", "tooltipText", "value")
_DETAIL_ECHO_KEYS = ("description", "hint", "stateDescription")


def _mask_node(node: dict, placeholder: str, key: str = "text") -> None:
    """Mask a node's piece and every other text that may echo it."""
    node[key] = placeholder
    for key in _ECHO_KEYS:
        if isinstance(node.get(key), str) and node[key]:
            node[key] = placeholder
    details = node.get("details")
    if isinstance(details, dict):
        for key in _DETAIL_ECHO_KEYS:
            if isinstance(details.get(key), str) and details[key]:
                details[key] = placeholder


def _collect_leaves(
    node: Any, out: list, in_input: bool, structured: bool, field: str = "text"
) -> None:
    """Ordered (leaf, is_input) pairs: nodes with their own ``field`` and no children.

    ``is_input`` is true for an input node or anything below one. iOS keeps
    typed text in ``value``, so both fields are scanned.
    """
    if isinstance(node, list):
        for item in node:
            _collect_leaves(item, out, in_input, structured, field)
    elif isinstance(node, dict):
        below = in_input or (structured and _is_input_node(node))
        children = node.get("children")
        if isinstance(children, list) and children:
            _collect_leaves(children, out, below, structured, field)
        elif isinstance(node.get(field), str) and node[field]:
            out.append((node, below))
        else:
            for key, value in node.items():
                if key not in _NON_TEXT_KEYS and isinstance(value, (list, dict)):
                    _collect_leaves(value, out, below, structured, field)


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
    """Per-run registry of secret values and the masking policy."""

    def __init__(self) -> None:
        self._entries: List[_Entry] = []
        self._lock = threading.Lock()
        self.black_screenshots = False
        self.mask_card_numbers = False
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
        """Register a value before it can reach observations or logs."""
        entry = self._entry(secret_id, value, redact)
        self._add([entry] if entry else [], protected)

    def register_partial(
        self,
        secret_id: str,
        value: str,
        *,
        redact: Literal["global", "input_only"] = "global",
    ) -> List[_Entry]:
        """Mask every prefix of ``value`` (typing may stop midway).

        One alternation per secret, longest prefix first, prefixes of at least
        four characters. A partial value can only sit in its field, so it is
        ``input_only``; only prefixes of eight or more digits keep the global
        digit rule. Returns the entries that were new.
        """
        prefixes: List[str] = []
        for k in range(len(value) - 1, _MIN_SEGMENTED_CHARS - 1, -1):
            prefix = value[:k].strip()
            if (
                prefix
                and len(re.sub(r"[\s\-]", "", prefix)) >= _MIN_SEGMENTED_CHARS
                and prefix not in prefixes
            ):
                prefixes.append(prefix)
        digits = _digits_of(value) is not None
        groups = [
            (
                [
                    p
                    for p in prefixes
                    if digits and len(_digits_of(p) or "") >= _MIN_GLOBAL_DIGITS
                ],
                True,
            ),
            (
                [
                    p
                    for p in prefixes
                    if not (digits and len(_digits_of(p) or "") >= _MIN_GLOBAL_DIGITS)
                ],
                False,
            ),
        ]
        entries = []
        for group, everywhere in groups:
            if not group:
                continue
            pattern = re.compile(
                "(?:" + "|".join(_compile(p)[0].pattern for p in group) + ")",
                re.IGNORECASE,
            )
            entries.append(
                self._build(secret_id, value, pattern, everywhere, True, True, group)
            )
        return self._add(entries, protected=False)

    def drop(self, entries: List[_Entry]) -> None:
        """Remove entries again (identity), e.g. prefixes after a good input."""
        with self._lock:
            self._entries = [
                e for e in self._entries if not any(e is d for d in entries)
            ]

    @staticmethod
    def _build(
        secret_id: str,
        value: str,
        pattern: Pattern[str],
        everywhere: bool,
        input_only: bool,
        partial: bool,
        samples: List[str],
    ) -> _Entry:
        compact = max(len(re.sub(r"[\s\-]", "", t)) for t in samples)
        return _Entry(
            secret_id=secret_id,
            pattern=pattern,
            everywhere=everywhere,
            input_only=input_only,
            segmentable=compact >= _MIN_SEGMENTED_CHARS,
            chars=frozenset(value.lower()) | {"-"},
            max_compact=compact,
            partial=partial,
        )

    def _entry(self, secret_id: str, value: str, redact: str) -> Optional[_Entry]:
        if not value or not value.strip():
            return None
        pattern, long_digits = _compile(value)
        return self._build(
            secret_id,
            value,
            pattern,
            long_digits,
            redact == "input_only",
            False,
            [value],
        )

    def _add(self, entries: List[_Entry], protected: bool) -> List[_Entry]:
        with self._lock:
            known = {e.key() for e in self._entries}
            added = []
            for entry in entries:
                if entry.key() not in known:
                    known.add(entry.key())
                    added.append(entry)
            # Copy-on-write: readers iterate a list that is never mutated.
            # Order secrets by their longest pattern. Only within one secret
            # must full values precede partial alternations to avoid suffix leaks.
            combined = [*self._entries, *added]
            lengths = {}
            for entry in combined:
                lengths[entry.secret_id] = max(
                    lengths.get(entry.secret_id, 0), len(entry.pattern.pattern)
                )
            self._entries = sorted(
                combined,
                key=lambda e: (
                    lengths[e.secret_id],
                    not e.partial,
                    len(e.pattern.pattern),
                ),
                reverse=True,
            )
            if protected:
                self.black_screenshots = True
            return added

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self.black_screenshots = False
            self.mask_card_numbers = False

    @property
    def active(self) -> bool:
        return bool(self._entries) or self.mask_card_numbers

    # -- masking -----------------------------------------------------------

    def redact_text(
        self, text: str, *, in_input: bool = False, structured: bool = False
    ) -> str:
        entries = self._entries
        if not self.active or not text:
            return text
        if self.mask_card_numbers:
            parts, end = [], 0
            for start, stop in _card_spans(text):
                parts.extend((text[end:start], PLACEHOLDER.format("card_number")))
                end = stop
            text = "".join([*parts, text[end:]])
        for entry in list(entries):
            if entry.applies(in_input, structured):
                text = entry.pattern.sub(PLACEHOLDER.format(entry.secret_id), text)
        return text

    def redact_data(
        self, data: Any, *, in_input: bool = False, structured: bool = False
    ) -> Any:
        """Return a copy of nested data with every string value masked."""
        if not self.active:
            return data
        masked = self._redact(data, in_input, structured)
        if isinstance(masked, (list, dict)):
            self._mask_segments(masked, in_input, structured)
        return masked

    def _redact(self, data: Any, in_input: bool, structured: bool) -> Any:
        if isinstance(data, str):
            return self.redact_text(data, in_input=in_input, structured=structured)
        if isinstance(data, list):
            return [self._redact(item, in_input, structured) for item in data]
        if isinstance(data, tuple):
            return tuple(self._redact(item, in_input, structured) for item in data)
        if isinstance(data, dict):
            node_input = in_input or (structured and _is_input_node(data))
            masked: dict = {}
            for key, value in data.items():
                if key in _SKIP_KEYS:
                    masked[key] = value
                    continue
                if key == "focusedElement":
                    child_in_input = True
                elif key == "children":
                    child_in_input = in_input
                else:
                    child_in_input = node_input
                masked[key] = self._redact(value, child_in_input, structured)
            return masked
        return data

    def _mask_segments(self, root: Any, in_input: bool, structured: bool) -> None:
        """Mask a value spread over consecutive leaf texts (``root`` is a copy).

        Slides a window of at most ``len(value)`` consecutive leaves over the
        leaves in tree order, so the cost is leaves x value length. Every
        contributing leaf may only use characters of the value; entries that
        do not apply outside inputs only look at input leaves. The card rule
        scans group-aligned windows, treating node boundaries as separators.
        """
        entries = [e for e in self._entries if e.segmentable]
        if not entries and not self.mask_card_numbers:
            return
        for field in ("text", "value"):
            leaves: list = []
            _collect_leaves(root, leaves, in_input, structured, field)
            if len(leaves) < 2:
                continue
            if self.mask_card_numbers:
                self._scan_card_runs(leaves, field)
            for entry in entries:
                restricted = not entry.applies(False, structured)
                run: list = []
                for node, eligible in [*leaves, (None, False)]:
                    if restricted and not eligible:
                        self._scan_run(entry, run, field)
                        run = []
                    elif node is not None:
                        run.append(node)
                self._scan_run(entry, run, field)

    @staticmethod
    def _scan_card_runs(leaves: list, field: str = "text") -> None:
        chunks, spans = [], []
        offset = 0
        for node, _ in leaves:
            text = node[field]
            chunks.append(text)
            spans.append((offset, offset + len(text)))
            offset += len(text) + 1
        index = 0
        for start, end in _card_spans(" ".join(chunks)):
            while index < len(spans) and spans[index][1] <= start:
                index += 1
            stop = index
            while stop < len(spans) and spans[stop][0] < end:
                stop += 1
            if stop - index >= 2:
                for node, _ in leaves[index:stop]:
                    _mask_node(node, PLACEHOLDER.format("card_number"), field)

    @staticmethod
    def _scan_run(entry: _Entry, run: list, field: str = "text") -> None:
        placeholder = PLACEHOLDER.format(entry.secret_id)
        for i in range(len(run)):
            window = []
            end = min(len(run), i + entry.max_compact)
            for index in range(i, end):
                node = run[index]
                if not entry.accepts(node[field]):
                    break
                window.append(node)
            if len(window) < 2:
                continue
            for separator in ("", " "):
                joined, spans = "", []
                for node in window:
                    spans.append((len(joined), len(joined) + len(node[field])))
                    joined += node[field] + separator
                for match in entry.pattern.finditer(joined):
                    hit = [
                        node
                        for node, (start, end) in zip(window, spans, strict=True)
                        if start < match.end() and end > match.start()
                    ]
                    if len(hit) >= 2:
                        for node in hit:
                            _mask_node(node, placeholder, field)

    def redact_ui_tree(self, data: Any) -> Any:
        """Redact the payload of ``driver.get_ui_tree()``."""
        if not self.active or not isinstance(data, dict):
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
        if not self.active or state is None:
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
    def log(self) -> Optional[list]:
        """Redacted view of the recording log; ``None`` if the chain has none."""
        raw = getattr(self.inner, "log", None)
        if not isinstance(raw, list):
            return None
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
        # Before the call: logs emitted while typing are masked too, and a
        # failed or interrupted input leaves the typed prefix covered.
        self.redactor.register(secret_id, value, redact=redact, protected=protected)
        prefixes = self.redactor.register_partial(secret_id, value, redact=redact)
        starts = self._log_lengths()
        typed = False
        try:
            typed = await self.inner.input_text(value, clear)
            return typed
        finally:
            if typed:
                self.redactor.drop(prefixes)
            self._scrub_recording_logs(starts, secret_id, value, protected)

    def _log_lengths(self) -> List[Tuple[list, int]]:
        """Every recording log in the wrapper chain with its current length."""
        found: List[Tuple[list, int]] = []
        seen: set = set()
        node = self.inner
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            attrs = vars(node) if hasattr(node, "__dict__") else {}
            log = attrs.get("log")
            if isinstance(log, list):
                found.append((log, len(log)))
            node = attrs.get("inner")
        return found

    @staticmethod
    def _scrub_recording_logs(
        starts: List[Tuple[list, int]], secret_id: str, value: str, protected: bool
    ) -> None:
        """Replace this call's plaintext ``input_text`` entries by a marker.

        Only entries appended since ``starts`` are touched: the value itself,
        or the words and blanks an inner Stealth wrapper typed one by one.
        """
        spoken = {value, " ", *value.split(" ")}
        for log, start in starts:
            hits = [
                i
                for i in range(start, len(log))
                if isinstance(log[i], dict)
                and log[i].get("action_type") == "input_text"
                and isinstance(log[i].get("text"), str)
                and log[i]["text"] in spoken
            ]
            if not hits:
                continue
            marker = {
                "action_type": "type_secret",
                "secret_id": secret_id,
                "clear": bool(log[hits[0]].get("clear", False)),
            }
            if protected:
                marker["protected"] = True
            log[hits[0]] = marker
            for i in reversed(hits[1:]):
                del log[i]


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
        # Providers that only offer get_state() serve final reads with it too.
        read = getattr(self.inner, "get_final_state", None) or self.inner.get_state
        state = await read(*args, **kwargs)
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
