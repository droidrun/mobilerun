import asyncio
import logging
from typing import Optional, Type, TypeVar

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    ToolCallBlock,
)
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel

from mobilerun.agent.utils.errors import describe_error
from mobilerun.agent.utils.errors import http_status_code as _http_status_code

logger = logging.getLogger("mobilerun")

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_HTTP_CLIENT_STATUS_CODES = {408, 409, 425, 429}


_BLOCKED = {
    "safety",
    "recitation",
    "blocklist",
    "prohibited_content",
    "content_filter",
    "refusal",
}
_TRUNCATED = {"max_tokens", "length", "max_output_tokens"}


def _as_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    try:
        dumped = dump() if callable(dump) else None
        return dumped if isinstance(dumped, dict) else {}
    except Exception:
        return {}


def _first_choice(mapping: dict, key: str) -> dict:
    value = mapping.get(key)
    if isinstance(value, list) and value:
        return _as_dict(value[0])
    return {}


def _empty_response_diagnostics(
    response: object, llm: object, *, stream: bool = False
) -> dict[str, object]:
    """Redacted metadata (types, key names, scalars) for an unusable response. Never content."""
    message = getattr(response, "message", None)
    raw = _as_dict(getattr(response, "raw", None))
    extra = _as_dict(getattr(response, "additional_kwargs", None))
    message_extra = _as_dict(getattr(message, "additional_kwargs", None))
    # Gemini OAuth stores the generateContent payload under raw["response"].
    nested = _as_dict(raw.get("response"))
    # finish_reason lives at top level, under LiteLLM choices[0], Gemini
    # candidates[0], or the nested Code Assist envelope.
    sources = [
        raw,
        extra,
        message_extra,
        nested,
        _first_choice(raw, "choices"),
        _first_choice(raw, "candidates"),
        _first_choice(nested, "choices"),
        _first_choice(nested, "candidates"),
    ]

    def scalar(
        *names: str, types: tuple[type, ...] = (str, int, float, bool)
    ) -> object | None:
        for src in sources:
            for name in names:
                value = src.get(name)
                if isinstance(value, types) and not (
                    isinstance(value, bool) and bool not in types
                ):
                    return value
        return None

    blocks = [type(b).__name__ for b in getattr(message, "blocks", None) or []]
    content_blocks = message_extra.get("content_blocks")
    block_types = [
        b["type"]
        for b in (content_blocks if isinstance(content_blocks, list) else [])
        if isinstance(b, dict) and isinstance(b.get("type"), str)
    ]
    tool_calls = bool(
        getattr(message, "tool_calls", None)
        or extra.get("tool_calls")
        or message_extra.get("tool_calls")
        or "ToolCallBlock" in blocks
        or "tool_use" in block_types
    )
    thinking = bool(
        any("think" in b.lower() for b in blocks)
        or message_extra.get("thinking")
        or "thinking" in block_types
    )
    # OpenAI Responses reports truncation as status="incomplete" plus
    # incomplete_details.reason instead of a finish_reason.
    incomplete = _as_dict(raw.get("incomplete_details")) or _as_dict(
        nested.get("incomplete_details")
    )
    incomplete_reason = incomplete.get("reason")
    reason = str(
        scalar("finish_reason", "finishReason", "stop_reason", "stopReason")
        or (incomplete_reason if isinstance(incomplete_reason, str) else "")
    ).lower()

    if response is None:
        category = "no_response"
    elif message is None and not hasattr(response, "text"):
        category = "no_message"
    elif reason in _BLOCKED:
        category = "provider_blocked"
    elif reason in _TRUNCATED:
        category = "truncated"
    elif tool_calls:
        category = "tool_calls_only"
    elif thinking:
        category = "thinking_only"
    elif any(b != "TextBlock" for b in blocks):
        category = "blocks_without_text"
    else:
        category = "empty_stream" if stream else "empty_content"

    return {
        "category": category,
        "stream": stream,
        "model": getattr(getattr(llm, "metadata", None), "model_name", None)
        or getattr(llm, "model", None),
        "response_type": type(response).__name__ if response is not None else None,
        "finish_reason": reason or None,
        "block_types": blocks,
        "has_tool_calls": tool_calls,
        "provider_request_id": scalar("request_id", "requestId", "id"),
        # Responses "status" is a string like "incomplete", not an HTTP code.
        "http_status": scalar("status_code", "statusCode", "status", types=(int,)),
        "raw_keys": sorted(map(str, raw)),
        "additional_kwargs_keys": sorted(map(str, extra)),
    }


def _log_empty_response(
    response: object, llm: object, attempt: int, *, stream: bool = False
) -> None:
    try:
        logger.warning(
            "LLM response unusable: attempt=%s diagnostics=%s",
            attempt,
            _empty_response_diagnostics(response, llm, stream=stream),
        )
    except Exception:
        # Diagnostics must never turn a retryable response into a hard error.
        logger.warning("LLM response unusable: diagnostics unavailable")


def _is_permanent_http_client_error(error: Exception) -> bool:
    """Return whether an HTTP client error should fail without another attempt."""
    status_code = _http_status_code(error)
    return (
        status_code is not None
        and 400 <= status_code < 500
        and status_code not in _RETRYABLE_HTTP_CLIENT_STATUS_CODES
    )


async def acall_with_retries(
    llm,
    messages: list,
    retries: int = 3,
    timeout: float = 500,
    delay: float = 1.0,
    stream: bool = False,
    **chat_kwargs,
) -> ChatResponse:
    """
    Call LLM with retries and timeout handling.

    Args:
        llm: The LLM client instance
        messages: List of messages to send
        retries: Number of retry attempts
        timeout: Timeout in seconds for each attempt
        delay: Base delay between retries (multiplied by attempt number)
        stream: If True, stream response chunks to console in real-time
        chat_kwargs: Extra request arguments, such as native tool definitions

    Returns:
        The LLM ChatResponse object
    """
    last_exception: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            if stream:
                response = await _stream_response(llm, messages, timeout, **chat_kwargs)
            else:
                response = await asyncio.wait_for(
                    llm.achat(messages=messages, **chat_kwargs),
                    timeout=timeout,
                )

            # A native tool call may have no text only when tools were sent.
            message = getattr(response, "message", None)
            if message is not None and (
                getattr(message, "content", None)
                or (
                    chat_kwargs.get("tools")
                    and any(
                        isinstance(b, ToolCallBlock)
                        for b in getattr(message, "blocks", [])
                    )
                )
            ):
                if not stream and message.content:
                    logger.info(f"{message.content}")
                return response
            else:
                _log_empty_response(response, llm, attempt, stream=stream)
                last_exception = ValueError("Empty response content")

        except asyncio.TimeoutError:
            logger.warning(f"Attempt {attempt} timed out after {timeout} seconds")
            last_exception = TimeoutError("Timed out")

        except Exception as e:
            logger.warning(
                f"Attempt {attempt} failed with error: "
                f"{type(e).__name__}: {describe_error(e)}"
            )
            if _is_permanent_http_client_error(e):
                raise
            last_exception = e

        if attempt < retries:
            await asyncio.sleep(delay * attempt)

    if last_exception:
        raise last_exception
    raise ValueError("All attempts returned empty response content")


async def _stream_response(
    llm, messages: list, timeout: float, **chat_kwargs
) -> ChatResponse:
    """
    Stream LLM response chunks to console and return accumulated response.

    Args:
        llm: The LLM client instance
        messages: List of messages to send
        timeout: Timeout in seconds for the entire stream

    Returns:
        ChatResponse with accumulated content
    """
    content = ""
    last_chunk: Optional[ChatResponse] = None

    async def stream_chunks():
        nonlocal content, last_chunk
        async for chunk in await llm.astream_chat(messages=messages, **chat_kwargs):
            delta = chunk.delta or ""
            if delta:
                logger.info(delta, extra={"stream": True})
            content += delta
            last_chunk = chunk
        logger.info("", extra={"stream_end": True})

    await asyncio.wait_for(stream_chunks(), timeout=timeout)

    # Build response matching non-streaming format
    # Use last_chunk.message to preserve all blocks (ThinkingBlock, etc.)
    # that providers accumulate during streaming
    response = ChatResponse(
        message=(
            last_chunk.message
            if last_chunk
            else ChatMessage(role="assistant", content=content)
        ),
        raw=last_chunk.raw if last_chunk else None,
        additional_kwargs=last_chunk.additional_kwargs if last_chunk else {},
    )

    return response


async def acomplete_with_retries(
    llm,
    prompt: str,
    retries: int = 3,
    timeout: float = 500,
    delay: float = 1.0,
    stream: bool = False,
) -> CompletionResponse:
    """
    Call LLM completion with retries and timeout handling.

    Args:
        llm: The LLM client instance
        prompt: The prompt string to send
        retries: Number of retry attempts
        timeout: Timeout in seconds for each attempt
        delay: Base delay between retries (multiplied by attempt number)
        stream: If True, stream response chunks to console in real-time

    Returns:
        The LLM CompletionResponse object
    """
    last_exception: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            if stream:
                response = await _stream_complete_response(llm, prompt, timeout)
            else:
                response = await asyncio.wait_for(
                    llm.acomplete(prompt),
                    timeout=timeout,
                )

            # Validate response
            if response is not None and getattr(response, "text", None):
                if not stream:
                    logger.info(f"{response.text}")
                return response
            else:
                _log_empty_response(response, llm, attempt, stream=stream)
                last_exception = ValueError("Empty response content")

        except asyncio.TimeoutError:
            logger.warning(f"Attempt {attempt} timed out after {timeout} seconds")
            last_exception = TimeoutError("Timed out")

        except Exception as e:
            logger.warning(
                f"Attempt {attempt} failed with error: "
                f"{type(e).__name__}: {describe_error(e)}"
            )
            if _is_permanent_http_client_error(e):
                raise
            last_exception = e

        if attempt < retries:
            await asyncio.sleep(delay * attempt)

    if last_exception:
        raise last_exception
    raise ValueError("All attempts returned empty response content")


async def _stream_complete_response(
    llm, prompt: str, timeout: float
) -> CompletionResponse:
    """
    Stream LLM completion response chunks to console and return accumulated response.

    Args:
        llm: The LLM client instance
        prompt: The prompt string to send
        timeout: Timeout in seconds for the entire stream

    Returns:
        CompletionResponse with accumulated content
    """
    content = ""
    last_chunk: Optional[CompletionResponse] = None

    async def stream_chunks():
        nonlocal content, last_chunk
        async for chunk in await llm.astream_complete(prompt):
            delta = chunk.delta or ""
            if delta:
                logger.info(delta, extra={"stream": True})
            content += delta
            last_chunk = chunk
        logger.info("", extra={"stream_end": True})

    await asyncio.wait_for(stream_chunks(), timeout=timeout)

    # Build response matching non-streaming format
    response = CompletionResponse(
        text=content,
        raw=last_chunk.raw if last_chunk else None,
        additional_kwargs=last_chunk.additional_kwargs if last_chunk else {},
    )

    return response


async def astructured_predict_with_retries(
    llm,
    output_cls: Type[T],
    prompt: PromptTemplate,
    retries: int = 3,
    timeout: float = 500,
    delay: float = 1.0,
    **prompt_args,
) -> T:
    """
    Call LLM structured predict with retries and timeout handling.

    Args:
        llm: The LLM client instance
        output_cls: The Pydantic model class for structured output
        prompt: PromptTemplate with {variables}
        retries: Number of retry attempts
        timeout: Timeout in seconds for each attempt
        delay: Base delay between retries (multiplied by attempt number)
        **prompt_args: Values for template variables

    Returns:
        Instance of the output_cls Pydantic model
    """
    last_exception: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            result = await asyncio.wait_for(
                llm.astructured_predict(output_cls, prompt, **prompt_args),
                timeout=timeout,
            )

            # Validate response
            if result is not None:
                logger.info(f"{result}")
                return result
            else:
                logger.warning(f"Attempt {attempt} returned None")
                _log_empty_response(None, llm, attempt)
                last_exception = ValueError("Empty response")

        except asyncio.TimeoutError:
            logger.warning(f"Attempt {attempt} timed out after {timeout} seconds")
            last_exception = TimeoutError("Timed out")

        except Exception as e:
            logger.warning(
                f"Attempt {attempt} failed with error: "
                f"{type(e).__name__}: {describe_error(e)}"
            )
            if _is_permanent_http_client_error(e):
                raise
            last_exception = e

        if attempt < retries:
            await asyncio.sleep(delay * attempt)

    if last_exception:
        raise last_exception
    raise ValueError("All attempts returned empty response")
