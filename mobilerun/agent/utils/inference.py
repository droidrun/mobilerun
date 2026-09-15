import asyncio
import logging
from typing import Optional, Type, TypeVar

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel

logger = logging.getLogger("mobilerun")

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_HTTP_CLIENT_STATUS_CODES = {408, 409, 425, 429}


_BLOCKED_FINISH_REASONS = {
    "safety",
    "recitation",
    "blocklist",
    "prohibited_content",
    "spii",
    "content_filter",
    "refusal",
}
_TRUNCATED_FINISH_REASONS = {"max_tokens", "length"}


def _empty_response_category(
    *,
    response: object,
    message: object,
    is_completion: bool,
    content: object,
    non_text_block_count: int,
    has_tool_calls: bool,
    has_thinking: bool,
    finish_reason: object,
    stop_reason: object,
    stream: bool,
) -> str:
    """Classify why a provider response carried no usable text content."""
    if response is None:
        return "no_response"
    if message is None and not is_completion:
        return "no_message"
    reason = str(finish_reason or stop_reason or "").lower()
    if reason in _BLOCKED_FINISH_REASONS:
        return "provider_blocked"
    if reason in _TRUNCATED_FINISH_REASONS:
        return "truncated"
    if has_tool_calls:
        return "tool_calls_only"
    if has_thinking:
        return "thinking_only"
    if non_text_block_count > 0:
        return "blocks_without_text"
    if stream:
        return "empty_stream"
    return "empty_content"


def _as_metadata_dict(value: object) -> dict | None:
    """Coerce a provider payload (dict or pydantic model) into a plain dict, else None."""
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump()
        except Exception:
            return None
        return dumped if isinstance(dumped, dict) else None
    return None


def _empty_response_diagnostics(
    response: object, llm: object, *, stream: bool = False
) -> dict[str, object]:
    """Return redacted metadata for an unusable provider response.

    Only types, key names, and scalar provider metadata are returned. Prompt text,
    message content, and full raw payloads are never included.
    """
    message = getattr(response, "message", None) if response is not None else None
    additional = (
        getattr(response, "additional_kwargs", None) if response is not None else None
    )
    raw = _as_metadata_dict(
        getattr(response, "raw", None) if response is not None else None
    )
    # CompletionResponse carries `text` directly instead of a message.
    is_completion = message is None and hasattr(response, "text")
    if is_completion:
        content = getattr(response, "text", None)
    else:
        content = getattr(message, "content", None)
    blocks = getattr(message, "blocks", None) or getattr(response, "blocks", None)
    metadata = getattr(llm, "metadata", None)
    model = getattr(metadata, "model_name", None) or getattr(llm, "model", None)
    if isinstance(blocks, (list, tuple)):
        block_types = [type(block).__name__ for block in blocks]
    else:
        block_types = []
    if isinstance(additional, dict):
        additional_keys = sorted(str(key) for key in additional)
    else:
        additional_keys = []
    if isinstance(raw, dict):
        raw_keys = sorted(str(key) for key in raw)
    else:
        raw_keys = []
    metadata_sources = [value for value in (raw, additional) if isinstance(value, dict)]
    # OpenAI/LiteLLM nest finish_reason under choices[0]; Gemini under candidates[0].
    for container_key in ("choices", "candidates"):
        entries = raw.get(container_key) if isinstance(raw, dict) else None
        first = (
            _as_metadata_dict(entries[0])
            if isinstance(entries, list) and entries
            else None
        )
        if first is not None:
            metadata_sources.append(first)

    def scalar_meta(*names: str) -> object | None:
        for source in metadata_sources:
            for name in names:
                value = source.get(name)
                if isinstance(value, (str, int, float, bool)):
                    return value
        return None

    has_tool_calls = bool(getattr(message, "tool_calls", None)) or bool(
        isinstance(additional, dict) and additional.get("tool_calls")
    )
    has_thinking = any("think" in name.lower() for name in block_types)
    non_text_block_count = sum(1 for name in block_types if name != "TextBlock")
    finish_reason = scalar_meta("finish_reason", "finishReason")
    stop_reason = scalar_meta("stop_reason", "stopReason")

    return {
        "category": _empty_response_category(
            response=response,
            message=message,
            is_completion=is_completion,
            content=content,
            non_text_block_count=non_text_block_count,
            has_tool_calls=has_tool_calls,
            has_thinking=has_thinking,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
            stream=stream,
        ),
        "stream": stream,
        "model": model,
        "response_type": type(response).__name__ if response is not None else None,
        "message_type": type(message).__name__ if message is not None else None,
        "content_type": type(content).__name__ if content is not None else None,
        "content_empty": not bool(content),
        "block_count": len(block_types),
        "block_types": block_types,
        "has_tool_calls": has_tool_calls,
        "has_thinking": has_thinking,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "provider_request_id": scalar_meta("request_id", "requestId", "id"),
        "http_status": scalar_meta("status_code", "statusCode", "status"),
        "additional_kwargs_keys": additional_keys,
        "raw_keys": raw_keys,
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
        # Diagnostics must never turn a recoverable provider response into a task error.
        logger.warning("LLM response unusable: diagnostics unavailable")


def _http_status_code(error: Exception) -> int | None:
    def read_attribute(value: object, name: str) -> object | None:
        try:
            return getattr(value, name, None)
        except Exception:
            return None

    def parse_status(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped.isascii() or not stripped.isdecimal():
                return None
            parsed = int(stripped)
        else:
            return None
        return parsed if 100 <= parsed <= 599 else None

    response = read_attribute(error, "response")
    candidates = (
        read_attribute(error, "status_code"),
        read_attribute(response, "status_code"),
        read_attribute(error, "code"),
        read_attribute(error, "status"),
        read_attribute(response, "status"),
    )
    for status_code in candidates:
        parsed = parse_status(status_code)
        if parsed is not None:
            return parsed
    return None


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

    Returns:
        The LLM ChatResponse object
    """
    last_exception: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            if stream:
                response = await _stream_response(llm, messages, timeout)
            else:
                response = await asyncio.wait_for(
                    llm.achat(messages=messages),
                    timeout=timeout,
                )

            # Validate response
            if (
                response is not None
                and getattr(response, "message", None) is not None
                and getattr(response.message, "content", None)
            ):
                if not stream:
                    logger.info(f"{response.message.content}")
                return response
            else:
                logger.warning(f"Attempt {attempt} returned empty content")
                _log_empty_response(response, llm, attempt, stream=stream)
                last_exception = ValueError("Empty response content")

        except asyncio.TimeoutError:
            logger.warning(f"Attempt {attempt} timed out after {timeout} seconds")
            last_exception = TimeoutError("Timed out")

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed with error: {e!r}")
            if _is_permanent_http_client_error(e):
                raise
            last_exception = e

        if attempt < retries:
            await asyncio.sleep(delay * attempt)

    if last_exception:
        raise last_exception
    raise ValueError("All attempts returned empty response content")


async def _stream_response(llm, messages: list, timeout: float) -> ChatResponse:
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
        async for chunk in await llm.astream_chat(messages=messages):
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
                logger.warning(f"Attempt {attempt} returned empty content")
                _log_empty_response(response, llm, attempt, stream=stream)
                last_exception = ValueError("Empty response content")

        except asyncio.TimeoutError:
            logger.warning(f"Attempt {attempt} timed out after {timeout} seconds")
            last_exception = TimeoutError("Timed out")

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed with error: {e!r}")
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
            logger.warning(f"Attempt {attempt} failed with error: {e!r}")
            if _is_permanent_http_client_error(e):
                raise
            last_exception = e

        if attempt < retries:
            await asyncio.sleep(delay * attempt)

    if last_exception:
        raise last_exception
    raise ValueError("All attempts returned empty response")
