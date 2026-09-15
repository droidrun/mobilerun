import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    ThinkingBlock,
)
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel

from mobilerun.agent.utils.inference import (
    _empty_response_diagnostics,
    _http_status_code,
    _log_empty_response,
    acall_with_retries,
    acomplete_with_retries,
    astructured_predict_with_retries,
)


def test_empty_response_diagnostics_are_metadata_only() -> None:
    response = SimpleNamespace(
        message=SimpleNamespace(
            content="",
            blocks=[SimpleNamespace(), SimpleNamespace()],
            tool_calls=[{"name": "tap"}],
        ),
        raw={"finish_reason": "SAFETY", "secret": "must-not-log"},
        additional_kwargs={"request_id": "req-1", "prompt": "secret"},
    )
    llm = SimpleNamespace(
        model="google/gemini-3.5-flash",
        metadata=SimpleNamespace(model_name="google/gemini-3.5-flash"),
    )

    diagnostics = _empty_response_diagnostics(response, llm)

    assert diagnostics["model"] == "google/gemini-3.5-flash"
    assert diagnostics["content_empty"] is True
    assert diagnostics["has_tool_calls"] is True
    assert diagnostics["finish_reason"] == "SAFETY"
    assert diagnostics["provider_request_id"] == "req-1"
    assert diagnostics["raw_keys"] == ["finish_reason", "secret"]
    assert "must-not-log" not in str(diagnostics)
    assert "prompt" in diagnostics["additional_kwargs_keys"]


def test_empty_response_diagnostics_never_leak_content() -> None:
    response = ChatResponse(
        message=ChatMessage(
            role="assistant",
            blocks=[ThinkingBlock(content="private reasoning")],
        ),
        raw={"candidates": [{"content": "leaked-candidate"}], "usage": {"total": 1}},
        additional_kwargs={"prompt_text": "user prompt goes here"},
    )

    diagnostics = _empty_response_diagnostics(response, SimpleNamespace())
    rendered = str(diagnostics)

    assert "private reasoning" not in rendered
    assert "leaked-candidate" not in rendered
    assert "user prompt goes here" not in rendered
    assert diagnostics["raw_keys"] == ["candidates", "usage"]
    assert diagnostics["additional_kwargs_keys"] == ["prompt_text"]


def test_empty_response_diagnostics_read_litellm_style_raw() -> None:
    class RawModel(BaseModel):
        id: str
        choices: list[dict]
        model: str

    raw = RawModel(
        id="chatcmpl-123",
        choices=[{"finish_reason": "content_filter", "message": {"content": "hidden"}}],
        model="gemini-3.5-flash",
    )
    response = ChatResponse(message=ChatMessage(role="assistant", content=""), raw=raw)

    diagnostics = _empty_response_diagnostics(response, SimpleNamespace())

    assert diagnostics["category"] == "provider_blocked"
    assert diagnostics["finish_reason"] == "content_filter"
    assert diagnostics["provider_request_id"] == "chatcmpl-123"
    assert diagnostics["raw_keys"] == ["choices", "id", "model"]
    assert "hidden" not in str(diagnostics)


def test_empty_response_diagnostics_read_gemini_candidates() -> None:
    response = ChatResponse(
        message=ChatMessage(role="assistant", content=""),
        raw={"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]},
    )

    diagnostics = _empty_response_diagnostics(response, SimpleNamespace())

    assert diagnostics["category"] == "truncated"
    assert diagnostics["finish_reason"] == "MAX_TOKENS"


def test_empty_response_category_tool_calls_in_additional_kwargs() -> None:
    response = ChatResponse(
        message=ChatMessage(
            role="assistant",
            content="",
            additional_kwargs={"tool_calls": [{"id": "c1"}]},
        ),
        additional_kwargs={"tool_calls": [{"id": "c1"}]},
    )

    diagnostics = _empty_response_diagnostics(response, SimpleNamespace())

    assert diagnostics["category"] == "tool_calls_only"
    assert diagnostics["has_tool_calls"] is True


def test_empty_response_category_no_response() -> None:
    diagnostics = _empty_response_diagnostics(None, SimpleNamespace(model="m"))

    assert diagnostics["category"] == "no_response"
    assert diagnostics["response_type"] is None
    assert diagnostics["model"] == "m"


def test_empty_response_category_completely_empty() -> None:
    response = ChatResponse(message=ChatMessage(role="assistant", content=""))

    diagnostics = _empty_response_diagnostics(response, SimpleNamespace())

    assert diagnostics["category"] == "empty_content"
    assert diagnostics["block_types"] == ["TextBlock"]
    assert diagnostics["finish_reason"] is None


@pytest.mark.parametrize(
    ("meta", "category"),
    [
        ({"finish_reason": "SAFETY"}, "provider_blocked"),
        ({"finish_reason": "RECITATION"}, "provider_blocked"),
        ({"finish_reason": "content_filter"}, "provider_blocked"),
        ({"finish_reason": "MAX_TOKENS"}, "truncated"),
        ({"stop_reason": "length"}, "truncated"),
    ],
)
def test_empty_response_category_from_finish_reason(meta: dict, category: str) -> None:
    response = ChatResponse(message=ChatMessage(role="assistant", content=""), raw=meta)

    diagnostics = _empty_response_diagnostics(response, SimpleNamespace())

    assert diagnostics["category"] == category
    assert (diagnostics["finish_reason"] or diagnostics["stop_reason"]) == next(
        iter(meta.values())
    )


def test_empty_response_category_thinking_only() -> None:
    response = ChatResponse(
        message=ChatMessage(
            role="assistant", blocks=[ThinkingBlock(content="thinking")]
        )
    )

    diagnostics = _empty_response_diagnostics(response, SimpleNamespace())

    assert diagnostics["category"] == "thinking_only"
    assert diagnostics["has_thinking"] is True
    assert diagnostics["block_types"] == ["ThinkingBlock"]


def test_empty_response_category_tool_calls_only() -> None:
    response = SimpleNamespace(
        message=SimpleNamespace(content="", blocks=[], tool_calls=[{"name": "tap"}]),
        raw={},
        additional_kwargs={},
    )

    diagnostics = _empty_response_diagnostics(response, SimpleNamespace())

    assert diagnostics["category"] == "tool_calls_only"


def test_empty_response_category_empty_stream() -> None:
    response = ChatResponse(message=ChatMessage(role="assistant", content=""))

    diagnostics = _empty_response_diagnostics(response, SimpleNamespace(), stream=True)

    assert diagnostics["category"] == "empty_stream"
    assert diagnostics["stream"] is True


@contextmanager
def _captured_logs() -> Iterator[list[logging.LogRecord]]:
    # The "mobilerun" logger does not propagate, so caplog never sees it.
    records: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("mobilerun")
    handler = Collector(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


class EmptyChatLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.model = "google/gemini-3.5-flash"

    async def achat(self, *, messages):
        self.calls += 1
        return ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            raw={"finish_reason": "SAFETY", "candidates": []},
        )

    async def astream_chat(self, *, messages):
        self.calls += 1

        async def chunks():
            if False:
                yield None

        return chunks()

    async def acomplete(self, prompt):
        self.calls += 1
        return CompletionResponse(text="", raw={"finish_reason": "MAX_TOKENS"})

    async def astructured_predict(self, output_cls, prompt, **prompt_args):
        self.calls += 1
        return None


def test_chat_empty_response_logs_diagnostics_and_keeps_retrying() -> None:
    llm = EmptyChatLLM()

    with _captured_logs() as records:
        with pytest.raises(ValueError, match="Empty response content"):
            asyncio.run(
                acall_with_retries(
                    llm, [{"role": "user", "content": "hello"}], retries=3, delay=0
                )
            )

    assert llm.calls == 3
    diagnostics_lines = [
        record.getMessage()
        for record in records
        if "LLM response unusable" in record.getMessage()
    ]
    assert len(diagnostics_lines) == 3
    assert "'category': 'provider_blocked'" in diagnostics_lines[0]
    assert "'finish_reason': 'SAFETY'" in diagnostics_lines[0]
    assert "'model': 'google/gemini-3.5-flash'" in diagnostics_lines[0]
    assert "hello" not in diagnostics_lines[0]


def test_chat_empty_stream_logs_stream_category() -> None:
    llm = EmptyChatLLM()

    with _captured_logs() as records:
        with pytest.raises(ValueError, match="Empty response content"):
            asyncio.run(
                acall_with_retries(
                    llm,
                    [{"role": "user", "content": "hello"}],
                    retries=2,
                    delay=0,
                    stream=True,
                )
            )

    diagnostics_lines = [
        record.getMessage()
        for record in records
        if "LLM response unusable" in record.getMessage()
    ]
    assert len(diagnostics_lines) == 2
    assert "'category': 'empty_stream'" in diagnostics_lines[0]
    assert "'stream': True" in diagnostics_lines[0]


def test_completion_empty_response_logs_truncated_category() -> None:
    llm = EmptyChatLLM()

    with _captured_logs() as records:
        with pytest.raises(ValueError, match="Empty response content"):
            asyncio.run(acomplete_with_retries(llm, "hello", retries=2, delay=0))

    diagnostics_lines = [
        record.getMessage()
        for record in records
        if "LLM response unusable" in record.getMessage()
    ]
    assert len(diagnostics_lines) == 2
    assert "'category': 'truncated'" in diagnostics_lines[0]


def test_structured_none_result_logs_no_response_category() -> None:
    llm = EmptyChatLLM()

    with _captured_logs() as records:
        with pytest.raises(ValueError, match="Empty response"):
            asyncio.run(
                astructured_predict_with_retries(
                    llm,
                    StructuredResult,
                    PromptTemplate("Return a value for {value}"),
                    retries=2,
                    delay=0,
                    value="hello",
                )
            )

    diagnostics_lines = [
        record.getMessage()
        for record in records
        if "LLM response unusable" in record.getMessage()
    ]
    assert len(diagnostics_lines) == 2
    assert "'category': 'no_response'" in diagnostics_lines[0]


def test_log_empty_response_never_raises() -> None:
    class Exploding:
        @property
        def message(self):
            raise RuntimeError("boom")

    with _captured_logs() as records:
        _log_empty_response(Exploding(), SimpleNamespace(), 1)

    assert any("diagnostics unavailable" in r.getMessage() for r in records)


class StatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class CodeError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.code = status_code


class ResponseStatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.response = SimpleNamespace(status=status_code)


class StructuredResult(BaseModel):
    value: str


class FailingLLM:
    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    async def achat(self, *, messages):
        self.calls += 1
        raise self.error

    async def acomplete(self, prompt):
        self.calls += 1
        raise self.error

    async def astructured_predict(self, output_cls, prompt, **prompt_args):
        self.calls += 1
        raise self.error


def _run_failing_helper(
    helper_name: str,
    status_code: int,
    error_type: type[Exception] = StatusError,
) -> int:
    llm = FailingLLM(error_type(status_code))

    with pytest.raises(error_type):
        if helper_name == "chat":
            asyncio.run(
                acall_with_retries(
                    llm,
                    [{"role": "user", "content": "hello"}],
                    retries=3,
                    delay=0,
                )
            )
        elif helper_name == "completion":
            asyncio.run(
                acomplete_with_retries(
                    llm,
                    "hello",
                    retries=3,
                    delay=0,
                )
            )
        else:
            asyncio.run(
                astructured_predict_with_retries(
                    llm,
                    StructuredResult,
                    PromptTemplate("Return a value for {value}"),
                    retries=3,
                    delay=0,
                    value="hello",
                )
            )

    return llm.calls


@pytest.mark.parametrize("helper_name", ["chat", "completion", "structured"])
@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_permanent_http_client_errors_are_not_retried(
    helper_name: str, status_code: int
) -> None:
    assert _run_failing_helper(helper_name, status_code) == 1


@pytest.mark.parametrize("helper_name", ["chat", "completion", "structured"])
@pytest.mark.parametrize("status_code", [408, 409, 425, 429, 500])
def test_transient_http_errors_are_retried(helper_name: str, status_code: int) -> None:
    assert _run_failing_helper(helper_name, status_code) == 3


def test_http_status_code_falls_back_to_exception_response() -> None:
    error = Exception("request failed")
    error.response = SimpleNamespace(status_code=401)

    assert _http_status_code(error) == 401


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (CodeError(403), 403),
        (ResponseStatusError(429), 429),
    ],
)
def test_http_status_code_supports_provider_specific_shapes(
    error: Exception, expected: int
) -> None:
    assert _http_status_code(error) == expected


def test_http_status_code_skips_malformed_candidates() -> None:
    error = Exception("request failed")
    error.status_code = False
    error.code = lambda: 403
    error.response = SimpleNamespace(status_code="not-a-status", status="401")

    assert _http_status_code(error) == 401


@pytest.mark.parametrize(
    "value",
    [True, False, 401.5, "401.0", "", "not-a-status", object(), 99, 600],
)
def test_http_status_code_rejects_invalid_values(value: object) -> None:
    error = Exception("request failed")
    error.code = value

    assert _http_status_code(error) is None


@pytest.mark.parametrize("helper_name", ["chat", "completion", "structured"])
@pytest.mark.parametrize("error_type", [CodeError, ResponseStatusError])
def test_provider_specific_permanent_http_errors_are_not_retried(
    helper_name: str, error_type: type[Exception]
) -> None:
    assert _run_failing_helper(helper_name, 403, error_type) == 1


@pytest.mark.parametrize("helper_name", ["chat", "completion", "structured"])
@pytest.mark.parametrize("error_type", [CodeError, ResponseStatusError])
def test_provider_specific_transient_http_errors_are_retried(
    helper_name: str, error_type: type[Exception]
) -> None:
    assert _run_failing_helper(helper_name, 429, error_type) == 3


def test_authentication_error_does_not_sleep_before_failing(monkeypatch) -> None:
    async def fail_if_called(delay):
        pytest.fail("permanent 401 errors must not enter retry backoff")

    monkeypatch.setattr(asyncio, "sleep", fail_if_called)

    assert _run_failing_helper("chat", 401) == 1
