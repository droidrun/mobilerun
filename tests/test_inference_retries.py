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
    ToolCallBlock,
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


class RawModel(BaseModel):
    id: str
    choices: list[dict]


def _chat(content: str = "", **kwargs) -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role="assistant", content=content), **kwargs
    )


@pytest.mark.parametrize(
    ("response", "stream", "category", "finish_reason"),
    [
        (None, False, "no_response", None),
        (_chat(), False, "empty_content", None),
        (_chat(), True, "empty_stream", None),
        (_chat(raw={"finish_reason": "SAFETY"}), False, "provider_blocked", "safety"),
        (
            _chat(raw={"candidates": [{"finishReason": "MAX_TOKENS"}]}),
            False,
            "truncated",
            "max_tokens",
        ),
        (
            _chat(raw=RawModel(id="r1", choices=[{"finish_reason": "content_filter"}])),
            False,
            "provider_blocked",
            "content_filter",
        ),
        (
            _chat(additional_kwargs={"tool_calls": [{"id": "c1"}]}),
            False,
            "tool_calls_only",
            None,
        ),
        (
            ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content="",
                    additional_kwargs={
                        "tool_calls": [{"id": "toolu_1", "name": "harmless"}]
                    },
                )
            ),
            False,
            "tool_calls_only",
            None,
        ),
        (
            ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    blocks=[
                        ToolCallBlock(tool_name="harmless", tool_call_id="c1"),
                    ],
                )
            ),
            False,
            "tool_calls_only",
            None,
        ),
        (
            ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content="",
                    additional_kwargs={
                        "content_blocks": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "harmless",
                                "input": {},
                            }
                        ]
                    },
                )
            ),
            False,
            "tool_calls_only",
            None,
        ),
        (
            ChatResponse(
                message=ChatMessage(
                    role="assistant", blocks=[ThinkingBlock(content="t")]
                )
            ),
            False,
            "thinking_only",
            None,
        ),
        (
            ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content="",
                    additional_kwargs={"thinking": {"type": "thinking"}},
                )
            ),
            False,
            "thinking_only",
            None,
        ),
        (
            CompletionResponse(text="", raw={"finish_reason": "length"}),
            False,
            "truncated",
            "length",
        ),
        (
            _chat(raw={"response": {"candidates": [{"finishReason": "MAX_TOKENS"}]}}),
            False,
            "truncated",
            "max_tokens",
        ),
    ],
)
def test_empty_response_category(response, stream, category, finish_reason) -> None:
    diagnostics = _empty_response_diagnostics(
        response, SimpleNamespace(), stream=stream
    )
    assert diagnostics["category"] == category
    assert diagnostics["finish_reason"] == finish_reason


def test_empty_response_diagnostics_never_leak_content() -> None:
    response = ChatResponse(
        message=ChatMessage(
            role="assistant",
            blocks=[ThinkingBlock(content="private reasoning")],
            additional_kwargs={
                "thinking": {"content": "hidden thought"},
                "tool_calls": [{"input": "secret arg"}],
            },
        ),
        raw=RawModel(id="req-1", choices=[{"message": {"content": "leaked"}}]),
        additional_kwargs={"prompt": "user prompt"},
    )
    diagnostics = _empty_response_diagnostics(response, SimpleNamespace(model="gemini"))
    rendered = str(diagnostics)

    assert diagnostics["model"] == "gemini"
    assert diagnostics["provider_request_id"] == "req-1"
    assert diagnostics["raw_keys"] == ["choices", "id"]
    assert diagnostics["additional_kwargs_keys"] == ["prompt"]
    for secret in (
        "private reasoning",
        "leaked",
        "user prompt",
        "hidden thought",
        "secret arg",
    ):
        assert secret not in rendered


@contextmanager
def _captured_logs() -> Iterator[list[str]]:
    # The "mobilerun" logger does not propagate, so caplog never sees it.
    messages: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())  # type: ignore[method-assign]
    logging.getLogger("mobilerun").addHandler(handler)
    try:
        yield messages
    finally:
        logging.getLogger("mobilerun").removeHandler(handler)


class EmptyLLM:
    calls = 0

    async def achat(self, *, messages):
        self.calls += 1
        return _chat(raw={"finish_reason": "SAFETY"})

    async def astream_chat(self, *, messages):
        self.calls += 1

        async def chunks():
            return
            yield

        return chunks()

    async def acomplete(self, prompt):
        self.calls += 1
        return CompletionResponse(text="")

    async def astructured_predict(self, output_cls, prompt, **prompt_args):
        self.calls += 1
        return None


@pytest.mark.parametrize(
    ("call", "category"),
    [
        (
            lambda llm: acall_with_retries(llm, ["hello"], retries=3, delay=0),
            "provider_blocked",
        ),
        (
            lambda llm: acall_with_retries(
                llm, ["hello"], retries=3, delay=0, stream=True
            ),
            "empty_stream",
        ),
        (
            lambda llm: acomplete_with_retries(llm, "hello", retries=3, delay=0),
            "empty_content",
        ),
        (
            lambda llm: astructured_predict_with_retries(
                llm,
                StructuredResult,
                PromptTemplate("{value}"),
                retries=3,
                delay=0,
                value="x",
            ),
            "no_response",
        ),
    ],
)
def test_empty_responses_log_category_and_still_retry(call, category) -> None:
    llm = EmptyLLM()
    with (
        _captured_logs() as messages,
        pytest.raises(ValueError, match="Empty response"),
    ):
        asyncio.run(call(llm))

    diagnostics = [m for m in messages if "LLM response unusable" in m]
    assert llm.calls == 3
    assert len(diagnostics) == 3
    assert f"'category': '{category}'" in diagnostics[0]
    assert "hello" not in diagnostics[0]


class ScriptedChatLLM:
    def __init__(self, response: ChatResponse) -> None:
        self.calls = 0
        self._response = response

    async def achat(self, *, messages):
        self.calls += 1
        return self._response


@pytest.mark.parametrize(
    ("response", "category"),
    [
        (
            ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content="",
                    additional_kwargs={
                        "tool_calls": [{"id": "toolu_1", "name": "harmless"}]
                    },
                )
            ),
            "tool_calls_only",
        ),
        (
            _chat(raw={"response": {"candidates": [{"finishReason": "MAX_TOKENS"}]}}),
            "truncated",
        ),
    ],
)
def test_adapter_empty_shapes_log_category_and_still_retry(response, category) -> None:
    llm = ScriptedChatLLM(response)
    with (
        _captured_logs() as messages,
        pytest.raises(ValueError, match="Empty response"),
    ):
        asyncio.run(acall_with_retries(llm, ["hello"], retries=3, delay=0))

    diagnostics = [m for m in messages if "LLM response unusable" in m]
    assert llm.calls == 3
    assert len(diagnostics) == 3
    assert f"'category': '{category}'" in diagnostics[0]
    assert "hello" not in diagnostics[0]


def test_log_empty_response_never_raises() -> None:
    class Exploding:
        @property
        def message(self):
            raise RuntimeError("boom")

    with _captured_logs() as messages:
        _log_empty_response(Exploding(), SimpleNamespace(), 1)
    assert any("diagnostics unavailable" in m for m in messages)


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
