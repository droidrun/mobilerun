"""Per-replay masking of credential values in normalized screen observations."""

import re
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from typing import Any


def _matches(pattern: re.Pattern[str], text: str) -> Iterator[tuple[int, int]]:
    return (match.span() for match in pattern.finditer(text))


def _bounded(pattern: str, sample: str) -> str:
    first, last = sample[0], sample[-1]
    if re.match(r"\w", first):
        pattern = (r"(?<!\d)" if first.isdigit() else r"(?<!\w)") + pattern
    if re.match(r"\w", last):
        pattern += r"(?!\d)" if last.isdigit() else r"(?!\w)"
    return pattern


_Edit = tuple[int, int, int, str]  # local start/end, full match length, marker


def _replace(text: str, spans: list[_Edit]) -> str:
    """Merge overlapping edits; the longest original match supplies the marker."""
    parts: list[str] = []
    end = 0
    best = 0
    for start, stop, length, marker in sorted(spans):
        if start >= end:
            parts.extend((text[end:start], marker))
            best = length
        elif length > best:
            parts[-1] = marker
            best = length
        end = max(end, stop)
    return "".join([*parts, text[end:]])


def _card_spans(text: str) -> Iterator[tuple[int, int]]:
    """Find group-aligned, Luhn-valid 13–19 digit numbers."""
    for run in re.finditer(r"[0-9]+(?:[\s-]+[0-9]+)*", text):
        groups = list(re.finditer(r"[0-9]+", run.group()))
        for i, first in enumerate(groups):
            digits = ""
            for last in groups[i : i + 19]:
                digits += last.group()
                if len(digits) > 19:
                    break
                if len(digits) >= 13:
                    terms = [
                        int(c) * (1 + index % 2)
                        for index, c in enumerate(reversed(digits))
                    ]
                    if sum(n - 9 if n > 9 else n for n in terms) % 10 == 0:
                        yield run.start() + first.start(), run.start() + last.end()


class CredentialMasks:
    """Mask only the current replay; never change host-owned observations."""

    def __init__(self) -> None:
        self._patterns: dict[tuple[str, str], re.Pattern[str]] = {}
        self.mask_card_numbers = False

    @property
    def active(self) -> bool:
        return bool(self._patterns) or self.mask_card_numbers

    def register(self, name: str, value: str) -> None:
        if not value.strip() or (name, value) in self._patterns:
            return
        digits = re.sub(r"[\s-]", "", value)
        if digits and digits.isdecimal():
            pattern = r"(?<!\d)" + r"[\s-]*".join(map(re.escape, digits)) + r"(?!\d)"
        else:
            tokens = re.findall(r"\s+|[^\s]+", value.strip())
            pattern = "".join(r"\s+" if t.isspace() else re.escape(t) for t in tokens)
            # UI providers often add whitespace around date separators.
            pattern = re.sub(r"(?<=\d)/(?=\d)", lambda _: r"\s*/\s*", pattern)
            pattern = _bounded(pattern, value.strip())
        self._patterns[name, value] = re.compile(pattern, re.IGNORECASE)

    def _rules(
        self,
    ) -> Iterator[tuple[str, Callable[[str], Iterable[tuple[int, int]]]]]:
        for (name, _), pattern in sorted(
            self._patterns.items(), key=lambda item: -len(item[0][1])
        ):
            yield f"[secret:{name}]", partial(_matches, pattern)
        if self.mask_card_numbers:
            yield "[secret:card_number]", _card_spans

    def snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        """Mask individual spans, including values spanning adjacent nodes."""
        pending: dict[tuple[int, str], list[_Edit]] = {}
        for marker, spans in self._rules():
            participants: set[int] = set()
            echoes: dict[str, int] = {}
            for key in ("text", "content_description"):
                nodes = [node for node in state["nodes"] if node.get(key)]
                texts = [node[key] for node in nodes]
                edits = [pending.setdefault((id(node), key), []) for node in nodes]
                for found, text in zip(edits, texts, strict=True):
                    found.extend((a, b, b - a, marker) for a, b in spans(text))
                if key == "content_description":
                    for i, node in enumerate(nodes):
                        if id(node) not in participants:
                            continue
                        for fragment, length in echoes.items():
                            edits[i].extend(
                                (m.start(), m.end(), length, marker)
                                for m in re.finditer(
                                    _bounded(re.escape(fragment), fragment),
                                    texts[i],
                                    re.IGNORECASE,
                                )
                            )
                for sep in ("", " "):
                    ends = []
                    offset = 0
                    for text in texts:
                        ends.append(offset + len(text))
                        offset = ends[-1] + len(sep)
                    for start, stop in spans(sep.join(texts)):
                        for i in range(bisect_right(ends, start), len(nodes)):
                            left = ends[i] - len(texts[i])
                            if left >= stop:
                                break
                            fragment_start = max(start - left, 0)
                            fragment_end = min(ends[i], stop) - left
                            edits[i].append(
                                (fragment_start, fragment_end, stop - start, marker)
                            )
                            if key == "text" and (start < left or stop > ends[i]):
                                fragment = texts[i][fragment_start:fragment_end]
                                if fragment.strip():
                                    echoes[fragment] = max(
                                        echoes.get(fragment, 0), stop - start
                                    )
                                    participants.add(id(nodes[i]))
            for data in [state["phone_state"], *state["nodes"]]:
                for key, value in data.items():
                    if key not in ("text", "content_description") and isinstance(
                        value, str
                    ):
                        pending.setdefault((id(data), key), []).extend(
                            (a, b, b - a, marker) for a, b in spans(value)
                        )
        for data in [state["phone_state"], *state["nodes"]]:
            for key, value in data.items():
                if edits_for_field := pending.get((id(data), key)):
                    data[key] = _replace(value, edits_for_field)
        return state


current_masks: ContextVar[CredentialMasks | None] = ContextVar(
    "credential_masks", default=None
)


@contextmanager
def masking_scope() -> Iterator[CredentialMasks]:
    masks = CredentialMasks()
    token = current_masks.set(masks)
    try:
        yield masks
    finally:
        masks._patterns.clear()
        current_masks.reset(token)
