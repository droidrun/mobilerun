"""Shared Anthropic model catalog and capability metadata."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

ANTHROPIC_API_DEFAULT_MODEL = "claude-sonnet-5"
ANTHROPIC_OAUTH_DEFAULT_MODEL = ANTHROPIC_API_DEFAULT_MODEL
ANTHROPIC_FABLE_5_1_MODEL = "claude-fable-5-1"
ANTHROPIC_OPUS_5_5_MODEL = "claude-opus-5-5"
ANTHROPIC_SONNET_5_5_MODEL = "claude-sonnet-5-5"

ANTHROPIC_API_MODELS = (
    ANTHROPIC_API_DEFAULT_MODEL,
    ANTHROPIC_SONNET_5_5_MODEL,
    ANTHROPIC_OPUS_5_5_MODEL,
    ANTHROPIC_FABLE_5_1_MODEL,
)

ANTHROPIC_OAUTH_MODELS = (
    ANTHROPIC_OAUTH_DEFAULT_MODEL,
    ANTHROPIC_SONNET_5_5_MODEL,
    ANTHROPIC_OPUS_5_5_MODEL,
    ANTHROPIC_FABLE_5_1_MODEL,
)

# Capability tables keep ids that left the catalogs so saved configs still work.
ANTHROPIC_MODEL_CONTEXT_WINDOWS = {
    ANTHROPIC_SONNET_5_5_MODEL: 1_000_000,
    ANTHROPIC_OPUS_5_5_MODEL: 1_000_000,
    "claude-opus-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    ANTHROPIC_FABLE_5_1_MODEL: 1_000_000,
    "claude-fable-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-haiku-4-5": 200_000,
}

# These models reject sampling controls. Filter them after all provider,
# profile, and per-request kwargs have been merged so an explicit override
# cannot accidentally restore an unsupported field.
ANTHROPIC_MODELS_WITHOUT_SAMPLING_PARAMS = frozenset(
    {
        ANTHROPIC_SONNET_5_5_MODEL,
        ANTHROPIC_OPUS_5_5_MODEL,
        "claude-opus-5",
        "claude-sonnet-5",
        ANTHROPIC_FABLE_5_1_MODEL,
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
    }
)
ANTHROPIC_UNSUPPORTED_SAMPLING_PARAMS = frozenset({"temperature", "top_p", "top_k"})

# These models reject forced tool_choice ("any"/"tool"), which function-based
# Pydantic programs send.
ANTHROPIC_MODELS_WITHOUT_FORCED_TOOL_CHOICE = frozenset(
    {ANTHROPIC_FABLE_5_1_MODEL, ANTHROPIC_OPUS_5_5_MODEL, ANTHROPIC_SONNET_5_5_MODEL}
)

# Anthropic models with the high-resolution visual-token budget
# (2576 px / 4784 tokens). Unknown ids remain on the conservative standard
# budget rather than assuming capabilities that have not been verified.
ANTHROPIC_HIGHRES_MODELS = frozenset(
    {
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        ANTHROPIC_OPUS_5_5_MODEL,
        ANTHROPIC_SONNET_5_5_MODEL,
        "claude-sonnet-5",
        ANTHROPIC_FABLE_5_1_MODEL,
        "claude-fable-5",
        "claude-mythos-5",
    }
)


def anthropic_model_context_window(model: object) -> int | None:
    """Return the verified context window for a known Anthropic model."""

    return ANTHROPIC_MODEL_CONTEXT_WINDOWS.get(str(model or "").strip())


def anthropic_model_omits_sampling_params(model: object) -> bool:
    """Whether the model rejects Anthropic sampling controls."""

    return str(model or "").strip() in ANTHROPIC_MODELS_WITHOUT_SAMPLING_PARAMS


def strip_anthropic_sampling_params(
    payload: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Remove unsupported sampling fields from a final Anthropic payload."""

    model = payload.get("model")
    if anthropic_model_omits_sampling_params(model):
        for param in ANTHROPIC_UNSUPPORTED_SAMPLING_PARAMS:
            payload.pop(param, None)
    return payload


class StreamUsage:
    """Usage of one Anthropic stream, attached to its latest chunk.

    The LlamaIndex adapter only yields content deltas, so the usage carried by
    ``message_start`` and ``message_delta`` never reaches callers. Usage is
    written onto the last yielded chunk before the stream ends, where both
    callers and LlamaIndex callbacks read it.
    """

    def __init__(self) -> None:
        self.usage: dict[str, Any] = {}
        self._chunk: Any = None

    def track(self, chunks: Any) -> Any:
        for chunk in chunks:
            self._chunk = chunk
            yield chunk

    async def atrack(self, chunks: Any) -> Any:
        async for chunk in chunks:
            self._chunk = chunk
            yield chunk

    def record(self, event: Any) -> None:
        kind = getattr(event, "type", None)
        if kind == "message_start":
            usage = getattr(getattr(event, "message", None), "usage", None)
        elif kind == "message_delta":
            usage = getattr(event, "usage", None)
        else:
            return
        if usage is None:
            return
        fields = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
        self.usage.update({k: v for k, v in fields.items() if v is not None})
        if self._chunk is not None:
            raw = self._chunk.raw if isinstance(self._chunk.raw, dict) else {}
            self._chunk.raw = {**raw, "usage": dict(self.usage)}


_stream_usage: ContextVar[StreamUsage | None] = ContextVar(
    "anthropic_stream_usage", default=None
)


@contextmanager
def collect_stream_usage(usage: StreamUsage) -> Iterator[None]:
    """Record the usage of Anthropic streams opened in this block."""

    token = _stream_usage.set(usage)
    try:
        yield
    finally:
        _stream_usage.reset(token)


def record_stream_usage(messages: Any, *, is_async: bool = False) -> None:
    """Wrap an SDK ``messages`` resource so streams report their usage."""

    create = messages.create

    def _wrap(stream: Any, usage: StreamUsage | None, kwargs: dict) -> Any:
        if usage is None or not kwargs.get("stream"):
            return stream
        return _UsageRecordingStream(stream, usage)

    if is_async:

        @wraps(create)
        async def acreate(*args: Any, **kwargs: Any) -> Any:
            usage = _stream_usage.get()
            return _wrap(await create(*args, **kwargs), usage, kwargs)

        messages.create = acreate
    else:

        @wraps(create)
        def screate(*args: Any, **kwargs: Any) -> Any:
            return _wrap(create(*args, **kwargs), _stream_usage.get(), kwargs)

        messages.create = screate


class _UsageRecordingStream:
    def __init__(self, stream: Any, usage: StreamUsage) -> None:
        self._stream = stream
        self._usage = usage

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __iter__(self) -> Any:
        for event in self._stream:
            self._usage.record(event)
            yield event

    async def __aiter__(self) -> Any:
        async for event in self._stream:
            self._usage.record(event)
            yield event
