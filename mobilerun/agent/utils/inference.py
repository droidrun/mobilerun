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
                last_exception = ValueError("Empty response content")

        except asyncio.TimeoutError:
            logger.warning(f"Attempt {attempt} timed out after {timeout} seconds")
            last_exception = TimeoutError("Timed out")

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed with error: {e!r}")
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
                last_exception = ValueError("Empty response content")

        except asyncio.TimeoutError:
            logger.warning(f"Attempt {attempt} timed out after {timeout} seconds")
            last_exception = TimeoutError("Timed out")

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed with error: {e!r}")
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
                last_exception = ValueError("Empty response")

        except asyncio.TimeoutError:
            logger.warning(f"Attempt {attempt} timed out after {timeout} seconds")
            last_exception = TimeoutError("Timed out")

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed with error: {e!r}")
            last_exception = e

        if attempt < retries:
            await asyncio.sleep(delay * attempt)

    if last_exception:
        raise last_exception
    raise ValueError("All attempts returned empty response")
