import asyncio
from types import SimpleNamespace

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ToolCallBlock
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel

from mobilerun.agent.utils.inference import (
    _http_status_code,
    acall_with_retries,
    acomplete_with_retries,
    astructured_predict_with_retries,
)


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


class ReplyLLM:
    def __init__(self, responses: list[ChatResponse]):
        self.responses = iter(responses)
        self.requests = []

    async def achat(self, *, messages, **kwargs):
        self.requests.append(kwargs)
        return next(self.responses)

    async def astream_chat(self, *, messages, **kwargs):
        response = await self.achat(messages=messages, **kwargs)

        async def chunks():
            yield response

        return chunks()


def _tool_only_response() -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(
            role="assistant",
            blocks=[
                ToolCallBlock(tool_call_id="call_1", tool_name="tap", tool_kwargs={})
            ],
        )
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("chat_kwargs", [{}, {"tools": []}, {"tools": None}])
def test_tool_only_reply_without_tools_is_retried(stream, chat_kwargs) -> None:
    text_response = ChatResponse(message=ChatMessage(role="assistant", content="hello"))
    llm = ReplyLLM([_tool_only_response(), text_response])

    response = asyncio.run(
        acall_with_retries(llm, [], retries=2, delay=0, stream=stream, **chat_kwargs)
    )

    assert response.message.content == "hello"
    assert llm.requests == [chat_kwargs, chat_kwargs]


@pytest.mark.parametrize("stream", [False, True])
def test_tool_only_reply_with_tools_is_returned_immediately(stream) -> None:
    tool_response = _tool_only_response()
    llm = ReplyLLM([tool_response])
    tools = [{"name": "tap", "parameters": {"type": "object", "properties": {}}}]

    response = asyncio.run(
        acall_with_retries(llm, [], delay=0, stream=stream, tools=tools)
    )

    assert response.message.content is None
    assert response.message.blocks == tool_response.message.blocks
    assert llm.requests == [{"tools": tools}]


@pytest.mark.parametrize("stream", [False, True])
def test_text_reply_is_returned_immediately(stream) -> None:
    llm = ReplyLLM(
        [ChatResponse(message=ChatMessage(role="assistant", content="hello"))]
    )

    response = asyncio.run(acall_with_retries(llm, [], delay=0, stream=stream))

    assert response.message.content == "hello"
    assert llm.requests == [{}]


@pytest.mark.parametrize("stream", [False, True])
def test_tool_only_replies_without_tools_exhaust_retries(stream) -> None:
    llm = ReplyLLM([_tool_only_response(), _tool_only_response()])

    with pytest.raises(ValueError, match="^Empty response content$"):
        asyncio.run(acall_with_retries(llm, [], retries=2, delay=0, stream=stream))

    assert llm.requests == [{}, {}]
