import asyncio
import logging

import httpx
import openai
import pytest
import requests

from mobilerun.agent.utils.errors import describe_error
from mobilerun.agent.utils.inference import acall_with_retries

CLOUDFLARE_PAGE = (
    "<html>\n  <head>\n    <title>Just a moment...</title>\n"
    + "    <script>window._cf_chl_opt={cvId: '3'};</script>\n" * 200
    + "  </head>\n</html>"
)
CODEX_URL = "https://chatgpt.com/backend-api/codex/chat/completions"


class _StatusError(Exception):
    """Stands in for SDK errors that expose the HTTP status (google-genai, etc.)."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _openai_error(body: str, status: int = 403) -> openai.APIStatusError:
    request = httpx.Request("POST", f"{CODEX_URL}?session=secret")
    response = httpx.Response(status, request=request, text=body)
    return openai.PermissionDeniedError(body, response=response, body=None)


def test_openai_html_error_page_is_summarized() -> None:
    assert describe_error(_openai_error(CLOUDFLARE_PAGE)) == (
        f"HTTP 403 HTML error page 'Just a moment...' from {CODEX_URL}"
    )


def test_doctype_page_with_status_prefix_keeps_the_prefix() -> None:
    error = _openai_error("Error code: 403 - <!DOCTYPE html><html><body>x</body>")

    assert describe_error(error) == f"Error code: 403 HTML error page from {CODEX_URL}"


def test_requests_html_error_is_summarized() -> None:
    response = requests.Response()
    response.status_code = 502
    response.url = "https://api.example.com/v1/messages?key=secret"
    error = requests.HTTPError("<html><title>Bad gateway</title></html>")
    error.response = response

    assert describe_error(error) == (
        "HTTP 502 HTML error page 'Bad gateway' from https://api.example.com/v1/messages"
    )


def test_long_messages_are_capped() -> None:
    text = describe_error(ValueError("x" * 2000), max_chars=100)

    assert text == "x" * 100 + "... [1900 more characters]"


@pytest.mark.parametrize(
    "message",
    ["Rate limit reached", "Error code: 400 - {'error': {'message': 'bad'}}"],
)
def test_short_messages_are_unchanged(message: str) -> None:
    assert describe_error(RuntimeError(message)) == message


def test_empty_messages_fall_back_to_the_error_type() -> None:
    assert describe_error(httpx.ConnectTimeout("")) == "ConnectTimeout"


@pytest.fixture
def mobilerun_caplog(caplog):
    """caplog wired to the non-propagating "mobilerun" logger."""
    logger = logging.getLogger("mobilerun")
    previous = logger.propagate
    logger.propagate = True
    caplog.set_level(logging.WARNING, logger="mobilerun")
    yield caplog
    logger.propagate = previous


def test_retry_warning_does_not_print_the_html_page(mobilerun_caplog) -> None:
    class FailingLLM:
        async def achat(self, messages):
            raise _openai_error(CLOUDFLARE_PAGE)

    with pytest.raises(openai.PermissionDeniedError):
        asyncio.run(acall_with_retries(FailingLLM(), [], retries=1, delay=0))

    [record] = [
        r for r in mobilerun_caplog.records if "Attempt 1 failed" in r.getMessage()
    ]
    assert record.getMessage() == (
        "Attempt 1 failed with error: PermissionDeniedError: "
        f"HTTP 403 HTML error page 'Just a moment...' from {CODEX_URL}"
    )


@pytest.mark.parametrize(
    "message",
    [
        "Failed to parse response: ```<html><body>hello</body></html>```",
        "Model output: <html><head><title>My page</title></head><body>hi</body></html>",
        "Failed to parse response: expected <action> tag but got: "
        "```<html><body>hello</body></html>```",
    ],
)
def test_quoted_html_is_not_treated_as_an_error_page(message: str) -> None:
    assert describe_error(ValueError(message)) == message


def test_quoted_html_in_an_http_error_is_not_treated_as_a_page() -> None:
    message = "Error code: 400 - Invalid content: ```<html><body>x</body></html>```"

    assert describe_error(_StatusError(message, 400)) == message


def test_url_credentials_and_query_are_dropped() -> None:
    request = httpx.Request(
        "POST", "http://proxyuser:proxypass@127.0.0.1:8443/v1/chat/completions?k=v"
    )
    response = httpx.Response(403, request=request, text=CLOUDFLARE_PAGE)
    error = openai.PermissionDeniedError(CLOUDFLARE_PAGE, response=response, body=None)

    text = describe_error(error)

    assert text.endswith("from http://127.0.0.1:8443/v1/chat/completions")
    assert "proxypass" not in text and "k=v" not in text


def test_wrapped_message_prefix_is_cleaned() -> None:
    error = _StatusError(
        "403 Forbidden. {'message': '<html><title>Denied</title>'}", 403
    )

    assert describe_error(error) == "403 Forbidden HTML error page 'Denied'"


def test_title_search_stays_fast_on_malformed_pages() -> None:
    import time

    started = time.perf_counter()
    text = describe_error(RuntimeError("<html>" + "<title>" * 30000))

    assert time.perf_counter() - started < 1
    assert text == "HTML error page"


def test_retry_log_filter_summarizes_html_pages() -> None:
    from mobilerun.agent.utils.errors import HtmlErrorPageFilter

    record = logging.LogRecord(
        "llama_index.llms.openai.utils",
        logging.WARNING,
        __file__,
        1,
        "Retrying %s in %s seconds as it raised %s.",
        ("OpenAI._achat", 0.5, f"InternalServerError: {CLOUDFLARE_PAGE}"),
        None,
    )

    assert HtmlErrorPageFilter().filter(record) is True
    assert record.getMessage() == (
        "Retrying OpenAI._achat in 0.5 seconds as it raised InternalServerError: "
        "HTML error page 'Just a moment...'"
    )


@pytest.mark.parametrize(
    "logger_name",
    ["llama_index.llms.openai.utils", "llama_index.llms.google_genai.utils"],
)
def test_retry_log_filter_is_installed_for_llama_index(logger_name: str) -> None:
    import mobilerun  # noqa: F401
    from mobilerun.agent.utils.errors import HtmlErrorPageFilter

    logger = logging.getLogger(logger_name)

    assert any(isinstance(f, HtmlErrorPageFilter) for f in logger.filters)


def test_tool_failure_summary_does_not_carry_the_html_page() -> None:
    from mobilerun.agent.tool_registry import ToolRegistry

    async def open_app(**kwargs):
        raise _openai_error(CLOUDFLARE_PAGE)

    registry = ToolRegistry()
    registry.register("open_app", open_app, params={}, description="Open an app")

    result = asyncio.run(registry.execute("open_app", {}, ctx=None))

    assert result.success is False
    assert result.summary == (
        "Failed to execute open_app: HTTP 403 HTML error page 'Just a moment...' "
        f"from {CODEX_URL}"
    )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            '<?xml version="1.0"?>\n<!-- gateway -->\n<!DOCTYPE html>'
            "<html><head><title>Bad gateway</title></head></html>",
            "HTML error page 'Bad gateway'",
        ),
        ("\ufeff<html><head><title>BOM</title></head>", "HTML error page 'BOM'"),
    ],
)
def test_pages_with_a_preamble_are_summarized(message: str, expected: str) -> None:
    assert describe_error(RuntimeError(message)) == expected


@pytest.mark.parametrize(
    ("message", "status", "expected"),
    [
        (
            "Error code: 502 - {'error': {'message': "
            "'<html><head><title>Oops</title></head>'}}",
            502,
            "Error code: 502 HTML error page 'Oops'",
        ),
        (
            "503 Service Unavailable. {'message': '\\n\\n  <html><head><title>T</title>'}",
            503,
            "503 Service Unavailable HTML error page 'T'",
        ),
    ],
)
def test_sdk_wrapped_pages_are_summarized(message, status, expected) -> None:
    assert describe_error(_StatusError(message, status)) == expected


def test_prefixed_page_without_http_status_is_left_as_is() -> None:
    message = "Error code: 502 - <html><head><title>Oops</title></head></html>"

    assert describe_error(RuntimeError(message)) == message


@pytest.mark.parametrize("message", ["Expected <html> root", "async <html> fail"])
def test_messages_that_only_mention_html_are_unchanged(message: str) -> None:
    assert describe_error(RuntimeError(message)) == message


def test_wrapped_capped_message_keeps_the_original_count() -> None:
    inner = describe_error(RuntimeError("x" * 5000))
    outer = describe_error(RuntimeError(f"Error calling LLM in executor: {inner}"))

    assert outer.endswith("... [3000 more characters]")


def test_google_genai_json_wrapped_page_keeps_only_the_status_text() -> None:
    from google.genai import errors as genai_errors

    page = "<html><head><title>Just a moment...</title></head><body>x</body></html>"
    error = genai_errors.ServerError(
        503, {"error": {"code": 503, "message": page, "status": "UNAVAILABLE"}}, None
    )

    assert describe_error(error) == "503 UNAVAILABLE HTML error page 'Just a moment...'"
