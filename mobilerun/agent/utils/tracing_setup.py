"""
Tracing setup utility for MobileAgent.

This module provides a centralized way to configure tracing providers
(Phoenix, Langfuse, etc.) based on the TracingConfig.
"""

import base64
import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import Token
from typing import Optional
from uuid import uuid4

import llama_index.core

from mobilerun.config_manager.config_manager import TracingConfig

logger = logging.getLogger("mobilerun")

_default_session_id: str = str(uuid4())
# Compatibility state for the deprecated standalone attachment helper only.
_legacy_session_context: tuple[str, str] = (_default_session_id, "anonymous")
_tracing_initialized: bool = False
_tracing_provider: Optional[str] = None
_langfuse_client: Optional[object] = None
_langfuse_preprocessor: Optional[object] = None
_langfuse_tracer_provider: Optional[object] = None
_langfuse_setup_attempted: bool = False
_tracing_setup_lock = threading.Lock()

DEFAULT_LANGFUSE_BASE_URL = "https://us.cloud.langfuse.com"


def setup_tracing(
    tracing_config: TracingConfig, agent: Optional[object] = None
) -> None:
    global _tracing_initialized, _tracing_provider, _legacy_session_context

    if not tracing_config.enabled:
        return

    provider = tracing_config.provider.lower()

    with _tracing_setup_lock:
        _legacy_session_context = (
            tracing_config.langfuse_session_id or _default_session_id,
            tracing_config.langfuse_user_id or "anonymous",
        )
        if _tracing_initialized:
            logger.debug(
                f"🔍 Tracing already initialized with {_tracing_provider}, skipping setup"
            )
            if provider == "langfuse" and agent:
                from mobilerun.telemetry.langfuse_processor import set_current_agent

                set_current_agent(agent)
            return

        if provider == "phoenix":
            if _setup_phoenix_tracing():
                _tracing_initialized = True
                _tracing_provider = "phoenix"
        elif provider == "langfuse":
            if _setup_langfuse_tracing(tracing_config, agent):
                _tracing_initialized = True
                _tracing_provider = "langfuse"
                logger.debug("🔍 Langfuse tracing enabled")
        else:
            logger.warning(
                f"⚠️  Unknown tracing provider: {provider}. "
                f"Supported providers: phoenix, langfuse"
            )


def _check_phoenix_reachable(endpoint: str, timeout: float = 3.0) -> bool:
    """Ping the Phoenix server to check if it's reachable."""
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(endpoint, method="GET")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _setup_phoenix_tracing() -> bool:
    """Set up Arize Phoenix tracing. Returns True if successful."""
    endpoint = os.getenv("PHOENIX_URL", "http://0.0.0.0:6006")
    # The handler imports its OpenTelemetry dependencies lazily, so a broken
    # install can surface while creating it as well as on import.
    try:
        from mobilerun.telemetry.phoenix import arize_phoenix_callback_handler

        if not _check_phoenix_reachable(endpoint):
            logger.warning(
                f"⚠️  Phoenix server is not reachable at {endpoint}. "
                "Tracing will be disabled for this session."
            )
            return False

        handler = arize_phoenix_callback_handler()
    except ImportError:
        logger.warning(
            "⚠️  Phoenix tracing dependencies are missing.\n"
            "    Reinstall Mobilerun to restore them:\n"
            "    • If installed via tool: `uv tool install --reinstall mobilerun`\n"
            "    • If installed via pip: `uv pip install --reinstall mobilerun`\n"
        )
        return False

    llama_index.core.global_handler = handler
    logger.debug("🔍 Arize Phoenix tracing enabled globally")
    return True


def _setup_langfuse_tracing(
    tracing_config: TracingConfig, agent: Optional[object] = None
) -> bool:
    """
    Set up Langfuse tracing with a preprocessor and the public v4 client.

    Args:
        tracing_config: TracingConfig instance containing Langfuse credentials
        agent: Optional MobileAgent instance to pass to span processor
    """

    global _langfuse_client
    global _langfuse_preprocessor, _langfuse_tracer_provider
    global _langfuse_setup_attempted

    try:
        from langfuse import Langfuse

        public_key = tracing_config.langfuse_public_key or os.getenv(
            "LANGFUSE_PUBLIC_KEY"
        )
        secret_key = tracing_config.langfuse_secret_key or os.getenv(
            "LANGFUSE_SECRET_KEY"
        )
        base_url = _resolve_langfuse_base_url(tracing_config)

        if not public_key or not secret_key:
            logger.error(
                "Langfuse credentials are missing. Configure both the public and secret key."
            )
            return False

        if _langfuse_setup_attempted:
            logger.error(
                "Langfuse tracing initialization already failed in this process; "
                "restart before retrying."
            )
            return False

        # STEP 1: Set up the tracer provider.
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        # Check if there's already a tracer provider (from Phoenix or previous setup)
        existing_provider = trace.get_tracer_provider()
        if hasattr(existing_provider, "add_span_processor"):
            # Use existing provider
            tracer_provider = existing_provider
            logger.debug("🔍 Using existing TracerProvider")
        else:
            # Create new provider
            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)
            logger.debug("🔍 Created new TracerProvider")

        if _provider_has_langfuse_processor(tracer_provider):
            logger.error(
                "Langfuse tracing is already registered on the active tracer provider; "
                "Mobilerun cannot guarantee preprocessing or export policy."
            )
            return False

        # STEP 2: Instrument LlamaIndex.
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        instrumentor = LlamaIndexInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
            logger.debug("🔍 Instrumented LlamaIndex")
        else:
            logger.debug("🔍 LlamaIndex already instrumented")

        # STEP 3: Patch the encoder once (now that instrumentation is active).
        from openinference.instrumentation.llama_index import _handler
        from pydantic import BaseModel as PydanticV2BaseModel

        if not getattr(_handler._encoder, "_mobilerun_pydantic_v2", False):
            original_encoder = _handler._encoder

            def _fixed_encoder(obj):
                """Encode Pydantic v2 models for OpenInference."""
                if isinstance(obj, PydanticV2BaseModel):
                    return obj.model_dump()
                return original_encoder(obj)

            _fixed_encoder._mobilerun_pydantic_v2 = True
            _handler._encoder = _fixed_encoder

        # STEP 4: Normalize private snapshots inside the SDK's processor chain.
        from mobilerun.telemetry.langfuse_processor import (
            LangfuseSpanProcessor,
            _LangfuseTracerProvider,
            set_current_agent,
        )

        if agent:
            set_current_agent(agent)

        if (
            _langfuse_preprocessor is None
            or _langfuse_tracer_provider is not tracer_provider
        ):
            _langfuse_preprocessor = LangfuseSpanProcessor()
            _langfuse_tracer_provider = tracer_provider

        # STEP 5: The public client owns the single exporter, queue, media uploads,
        # flushing, and shutdown. Export all spans so Mobilerun's custom workflow
        # and screenshot spans are not removed by Langfuse v4's default filter.
        # A failed SDK construction can leave processors attached to an OTel
        # provider, whose public API cannot remove them. Do not retry in-process:
        # that is the only general way to guarantee no duplicate exporters.
        _langfuse_setup_attempted = True
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            tracer_provider=_LangfuseTracerProvider(
                tracer_provider, _langfuse_preprocessor
            ),
            should_export_span=_export_all_spans,
        )

        if not _provider_has_owned_langfuse_pipeline(tracer_provider):
            logger.error(
                "Langfuse is already initialized on another tracer provider; "
                "Mobilerun cannot guarantee preprocessing or export policy."
            )
            return False

        _langfuse_client = client

        try:
            if not _langfuse_client.auth_check():
                logger.error(
                    "Langfuse authentication failed. Please check your credentials."
                )
        except Exception as error:
            logger.error(
                "Unable to verify Langfuse authentication; tracing remains configured."
            )
            logger.debug(
                "Langfuse authentication diagnostic failed (%s)",
                type(error).__name__,
            )

        return True

    except ImportError as e:
        logger.warning(
            "⚠️  Langfuse dependencies are not installed.\n"
            "    To enable Langfuse integration, install with:\n"
            "    • If installed via tool: `uv tool install mobilerun[langfuse]`\n"
            "    • If installed via pip: `uv pip install mobilerun[langfuse]`\n"
            f"    Missing: {e.name if hasattr(e, 'name') else str(e)}\n"
        )
        return False
    except Exception as error:
        logger.error("Failed to initialize Langfuse tracing (%s)", type(error).__name__)
        return False


def _resolve_langfuse_base_url(tracing_config: TracingConfig) -> str:
    """Resolve the Langfuse endpoint, keeping LANGFUSE_HOST as a fallback."""
    return (
        tracing_config.langfuse_host
        or os.getenv("LANGFUSE_BASE_URL")
        or os.getenv("LANGFUSE_HOST")
        or DEFAULT_LANGFUSE_BASE_URL
    )


def _export_all_spans(_span) -> bool:
    """Keep custom Mobilerun spans alongside standard LLM spans in Langfuse."""
    return True


def _provider_has_langfuse_processor(tracer_provider: object) -> bool:
    """Detect both directly registered and wrapped Langfuse processors."""
    from mobilerun.telemetry.langfuse_processor import _LangfuseSpanProcessor

    return any(
        isinstance(processor, _LangfuseSpanProcessor)
        or type(processor).__module__.startswith("langfuse.")
        for processor in _provider_span_processors(tracer_provider)
    )


def _provider_has_owned_langfuse_pipeline(tracer_provider: object) -> bool:
    """Confirm exactly one SDK processor uses our snapshot normalizer."""
    from mobilerun.telemetry.langfuse_processor import _LangfuseSpanProcessor

    processors = [
        processor
        for processor in _provider_span_processors(tracer_provider)
        if isinstance(processor, _LangfuseSpanProcessor)
        or type(processor).__module__.startswith("langfuse.")
    ]
    return (
        len(processors) == 1
        and isinstance(processors[0], _LangfuseSpanProcessor)
        and processors[0].normalizer is _langfuse_preprocessor
        and type(processors[0].processor).__module__.startswith("langfuse.")
    )


def _provider_span_processors(tracer_provider: object) -> tuple[object, ...]:
    """Read the active OTel pipeline for ordering validation."""
    active_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = getattr(active_processor, "_span_processors", None)
    if processors is None:
        processors = getattr(tracer_provider, "processors", ())
    return tuple(processors)


def _attach_session_context(session_id: Optional[str], user_id: Optional[str]) -> Token:
    """Attach explicit identifiers and return the token used to restore them."""
    from openinference.semconv.trace import SpanAttributes
    from opentelemetry.context import attach, get_current, set_value

    ctx = get_current()
    ctx = set_value(SpanAttributes.SESSION_ID, session_id or _default_session_id, ctx)
    ctx = set_value(SpanAttributes.USER_ID, user_id or "anonymous", ctx)
    return attach(ctx)


def apply_session_context() -> None:
    """Deprecated: use session_context for scoped tracing instead.

    Retains the legacy no-argument behavior: attach the last configured session
    and user identifiers without detaching them. MobileAgent does not use this
    compatibility state.
    """
    if not _tracing_initialized or _tracing_provider != "langfuse":
        return
    _attach_session_context(*_legacy_session_context)


@contextmanager
def session_context(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent: Optional[object] = None,
) -> Iterator[None]:
    """Scope Langfuse identifiers and agent metadata to workflow scheduling."""
    if not _tracing_initialized or _tracing_provider != "langfuse":
        yield
        return

    from opentelemetry.context import detach

    from mobilerun.telemetry.langfuse_processor import _current_agent

    token = _attach_session_context(session_id, user_id)
    agent_token = _current_agent.set(agent) if agent is not None else None
    try:
        yield
    finally:
        if agent_token is not None:
            _current_agent.reset(agent_token)
        detach(token)


def record_langfuse_screenshot(
    screenshot: bytes,
    mime_type: str = "image/png",
    parent_span=None,
    screenshots_enabled: bool = False,
    vision_enabled: bool = False,
) -> None:
    """
    Emit a tracing span that carries a screenshot for Langfuse uploads.

    Only active when Langfuse tracing is enabled and screenshots are enabled.
    """
    if (
        not _tracing_initialized
        or _tracing_provider != "langfuse"
        or not screenshot
        or not screenshots_enabled
        or vision_enabled  # avoid duplicate uploads when vision already embeds images in LLM spans
    ):
        return

    try:
        from opentelemetry import trace

        from mobilerun.telemetry.langfuse_processor import (
            get_last_step_span_context,
            get_root_span_context,
        )

        tracer = trace.get_tracer("droidrun.screenshot")
        image_b64 = base64.b64encode(screenshot).decode()

        # Attach to the provided span if valid; otherwise use current span; else root; skip if none.
        candidate = (
            parent_span
            if parent_span and parent_span.get_span_context().is_valid
            else None
        )
        if candidate is None:
            current_span = trace.get_current_span()
            if current_span and current_span.get_span_context().is_valid:
                candidate = current_span

        parent_ctx = (
            trace.set_span_in_context(candidate)
            if candidate is not None
            else (get_last_step_span_context() or get_root_span_context())
        )

        if parent_ctx is None:
            return

        span = tracer.start_span("droidrun.screenshot", context=parent_ctx)
        try:
            span.set_attribute("droidrun.screenshot.image_base64", image_b64)
            span.set_attribute("droidrun.screenshot.mime_type", mime_type)
        finally:
            span.end()
    except Exception as e:
        logger.debug(f"Failed to record Langfuse screenshot span: {e}")
