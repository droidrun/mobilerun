"""
Arize Phoenix tracing integration for Droidrun.

This module provides Phoenix instrumentation for tracing LLM calls and agent execution.
It includes utilities for creating custom spans with clean names and context management.
"""

import asyncio
import functools
import inspect
import os
import uuid
from contextvars import Token, copy_context
from typing import Any, Callable

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index_instrumentation import get_dispatcher
from openinference.instrumentation import TraceConfig

dispatcher = get_dispatcher()


def arize_phoenix_callback_handler(**kwargs: Any) -> BaseCallbackHandler:
    """
    Create and configure an Arize Phoenix callback handler for LlamaIndex.

    This function sets up OpenTelemetry tracing with Phoenix backend for monitoring
    LLM calls and agent execution.

    Args:
        **kwargs: Optional configuration overrides
            - endpoint: Phoenix server URL (default: http://0.0.0.0:6006 or PHOENIX_URL env var)
            - tracer_provider: Custom tracer provider
            - separate_trace_from_runtime_context: Separate traces from runtime context

    Returns:
        Configured LlamaIndex instrumentor instance

    Environment Variables:
        - PHOENIX_URL: Phoenix server URL
        - PHOENIX_PROJECT_NAME: Project name for organizing traces
    """
    # newer versions of arize, v2.x
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from openinference.semconv.resource import ResourceAttributes
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    endpoint = (
        kwargs.get("endpoint", os.getenv("phoenix_url", "http://0.0.0.0:6006"))
        + "/v1/traces"
    )

    resource_attributes = {}
    phoenix_project_name = os.getenv("phoenix_project_name", "")
    if phoenix_project_name.strip():
        resource_attributes[ResourceAttributes.PROJECT_NAME] = phoenix_project_name
    resource = Resource(attributes=resource_attributes)

    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
    config = TraceConfig(base64_image_max_length=64000000)

    return LlamaIndexInstrumentor().instrument(
        tracer_provider=kwargs.get("tracer_provider", tracer_provider),
        separate_trace_from_runtime_context=kwargs.get(
            "separate_trace_from_runtime_context"
        ),
        config=config,
    )


def clean_span(span_name: str):
    """
    Create a span with a clean name (without class prefix).

    This function returns a decorator that creates spans with custom names
    instead of the default class.method format.

    It preserves parent-child relationships by using the same active span
    context variable as the built-in dispatcher decorator does.

    Args:
        span_name: The desired name for the span

    Returns:
        A decorator function
    """

    def decorator(func: Callable) -> Callable:
        # Support both sync and async callables
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Import here to avoid circular imports
                from llama_index_instrumentation.dispatcher import (
                    active_instrument_tags,
                )
                from llama_index_instrumentation.span import active_span_id

                span_id = f"{span_name}-{uuid.uuid4()}"
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                # Treat as method only if qualname indicates Class.method
                is_method = "." in getattr(func, "__qualname__", "")
                instance = args[0] if (args and is_method) else None

                tags = active_instrument_tags.get()
                token = active_span_id.set(span_id)
                parent_id = (
                    None if token.old_value is Token.MISSING else token.old_value
                )

                dispatcher.span_enter(
                    id_=span_id,
                    bound_args=bound_args,
                    instance=instance,
                    parent_id=parent_id,
                    tags=tags,
                )
                try:
                    result = await func(*args, **kwargs)
                except Exception as e:
                    dispatcher.span_drop(
                        id_=span_id, bound_args=bound_args, instance=instance, err=e
                    )
                    raise
                else:
                    dispatcher.span_exit(
                        id_=span_id,
                        bound_args=bound_args,
                        instance=instance,
                        result=result,
                    )
                    return result
                finally:
                    active_span_id.reset(token)

            return async_wrapper
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Import here to avoid circular imports
                from llama_index_instrumentation.dispatcher import (
                    active_instrument_tags,
                )
                from llama_index_instrumentation.span import active_span_id

                span_id = f"{span_name}-{uuid.uuid4()}"
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                # Treat as method only if qualname indicates Class.method
                is_method = "." in getattr(func, "__qualname__", "")
                instance = args[0] if (args and is_method) else None

                tags = active_instrument_tags.get()
                context = copy_context()
                token = active_span_id.set(span_id)
                parent_id = (
                    None if token.old_value is Token.MISSING else token.old_value
                )

                dispatcher.span_enter(
                    id_=span_id,
                    bound_args=bound_args,
                    instance=instance,
                    parent_id=parent_id,
                    tags=tags,
                )
                try:
                    result = func(*args, **kwargs)
                    if isinstance(result, asyncio.Future):
                        new_future = asyncio.ensure_future(result)

                        def _on_done(fut: asyncio.Future) -> None:
                            try:
                                fut_result = None if fut.exception() else fut.result()
                                dispatcher.span_exit(
                                    id_=span_id,
                                    bound_args=bound_args,
                                    instance=instance,
                                    result=fut_result,
                                )
                            except Exception as e2:
                                dispatcher.span_drop(
                                    id_=span_id,
                                    bound_args=bound_args,
                                    instance=instance,
                                    err=e2,
                                )
                                raise
                            finally:
                                try:
                                    context.run(active_span_id.reset, token)
                                except ValueError:
                                    pass

                        new_future.add_done_callback(_on_done)
                        return new_future
                except Exception as e:
                    dispatcher.span_drop(
                        id_=span_id, bound_args=bound_args, instance=instance, err=e
                    )
                    raise
                else:
                    dispatcher.span_exit(
                        id_=span_id,
                        bound_args=bound_args,
                        instance=instance,
                        result=result,
                    )
                    return result
                finally:
                    if not isinstance(locals().get("result"), asyncio.Future):
                        active_span_id.reset(token)

            return wrapper

    return decorator
