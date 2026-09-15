import asyncio
from types import SimpleNamespace

import pytest
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel

from mobilerun.agent.utils.inference import (
    _empty_response_diagnostics,
    _http_status_code,
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
