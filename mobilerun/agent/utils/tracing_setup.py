"""
Tracing setup utility for DroidAgent.

This module provides a centralized way to configure tracing providers
(Phoenix, Langfuse, etc.) based on the TracingConfig.
"""

import logging
import os
from typing import Optional
from uuid import uuid4
import base64

import llama_index.core

from droidrun.config_manager.config_manager import TracingConfig

logger = logging.getLogger("droidrun")

_default_session_id: str = str(uuid4())
_session_id: str = _default_session_id
_tracing_initialized: bool = False
_tracing_provider: Optional[str] = None
_user_id: str = "anonymous"


def setup_tracing(
    tracing_config: TracingConfig, agent: Optional[object] = None
) -> None:
    global _tracing_initialized, _tracing_provider, _session_id, _user_id

    if not tracing_config.enabled:
        return

    provider = tracing_config.provider.lower()

    if tracing_config.langfuse_session_id:
        _session_id = tracing_config.langfuse_session_id
    else:
        _session_id = _default_session_id

    if tracing_config.langfuse_user_id:
        _user_id = tracing_config.langfuse_user_id
    else:
        _user_id = "anonymous"

    if _tracing_initialized:
        logger.debug(
            f"🔍 Tracing already initialized with {_tracing_provider}, skipping setup"
        )
        if provider == "langfuse" and agent:
            from droidrun.telemetry.langfuse_processor import set_current_agent

            set_current_agent(agent)
        return

    if provider == "phoenix":
        if _setup_phoenix_tracing():
            _tracing_initialized = True
            _tracing_provider = "phoenix"
    elif provider == "langfuse":
        _setup_langfuse_tracing(tracing_config, agent)
        _tracing_initialized = True
        _tracing_provider = "langfuse"
        logger.debug(f"🔍 Langfuse tracing enabled | Session: {_session_id}")
    else:
        logger.warning(
            f"⚠️  Unknown tracing provider: {provider}. "
            f"Supported providers: phoenix, langfuse"
        )


def _check_phoenix_reachable(endpoint: str, timeout: float = 3.0) -> bool:
    """Ping the Phoenix server to check if it's reachable."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(endpoint, method="GET")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _setup_phoenix_tracing() -> bool:
    """Set up Arize Phoenix tracing. Returns True if successful."""
    try:
        from droidrun.telemetry.phoenix import arize_phoenix_callback_handler
    except ImportError:
        logger.warning(
            "⚠️  Arize Phoenix is not installed.\n"
            "    To enable Phoenix integration, install with:\n"
            "    • If installed via tool: `uv tool install droidrun[phoenix]`"
            "    • If installed via pip: `uv pip install droidrun[phoenix]`\n"
        )
        return False

    endpoint = os.getenv("PHOENIX_URL", "http://0.0.0.0:6006")
    if not _check_phoenix_reachable(endpoint):
        logger.warning(
            f"⚠️  Phoenix server is not reachable at {endpoint}. "
            "Tracing will be disabled for this session."
        )
        return False

    handler = arize_phoenix_callback_handler()
    llama_index.core.global_handler = handler
    logger.debug("🔍 Arize Phoenix tracing enabled globally")
    return True


def _setup_langfuse_tracing(
    tracing_config: TracingConfig, agent: Optional[object] = None
) -> None:
    """
    Set up Langfuse tracing with custom span processor.

    Args:
        tracing_config: TracingConfig instance containing Langfuse credentials
        agent: Optional DroidAgent instance to pass to span processor
    """

    try:
        # Set environment variables
        if tracing_config.langfuse_secret_key:
            os.environ["LANGFUSE_SECRET_KEY"] = tracing_config.langfuse_secret_key
        if tracing_config.langfuse_public_key:
            os.environ["LANGFUSE_PUBLIC_KEY"] = tracing_config.langfuse_public_key
        if tracing_config.langfuse_host:
            os.environ["LANGFUSE_HOST"] = tracing_config.langfuse_host
        else:
            os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com"

        # Verify credentials
        from langfuse import Langfuse

        langfuse = Langfuse()
        try:
            if not langfuse.auth_check():
                logger.error(
                    "❌ Langfuse authentication failed. Please check your credentials."
                )
                return
        except Exception as e:
            logger.error(
                f"Error checking Langfuse authentication: {e}\nLikely a network issue or credentials are incorrect"
            )
            return

        # STEP 1: Set up tracer provider (before any LlamaIndex imports!)
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry import trace

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

        # STEP 2: Instrument LlamaIndex FIRST (before any LlamaIndex imports!)
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        instrumentor = LlamaIndexInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
            logger.debug("🔍 Instrumented LlamaIndex")
        else:
            logger.debug("🔍 LlamaIndex already instrumented")

        # STEP 3: Patch the encoder (now that instrumentation is active)
        from pydantic import BaseModel as PydanticV2BaseModel
        from openinference.instrumentation.llama_index import _handler

        _original_encoder = _handler._encoder

        def _fixed_encoder(obj):
            """Fixed encoder that properly handles Pydantic v2 models."""
            if isinstance(obj, PydanticV2BaseModel):
                return obj.model_dump()
            return _original_encoder(obj)

        _handler._encoder = _fixed_encoder

        # STEP 4: Add our custom processor (after instrumentation is set up)
        from droidrun.telemetry.langfuse_processor import (
            LangfuseSpanProcessor,
            set_current_agent,
        )

        if agent:
            set_current_agent(agent)

        span_processor = LangfuseSpanProcessor(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            base_url=os.environ["LANGFUSE_HOST"],
        )
        tracer_provider.add_span_processor(span_processor)

    except ImportError as e:
        logger.warning(
            "⚠️  Langfuse dependencies are not installed.\n"
            "    To enable Langfuse integration, install with:\n"
            "    • If installed via tool: `uv tool install droidrun[langfuse]`\n"
            "    • If installed via pip: `uv pip install droidrun[langfuse]`\n"
            f"    Missing: {e.name if hasattr(e, 'name') else str(e)}\n"
        )


def apply_session_context() -> None:
    """Apply session context for tracing. Only active when Langfuse tracing is enabled."""
    if not _tracing_initialized or _tracing_provider != "langfuse":
        return

    from opentelemetry.context import attach, get_current, set_value
    from openinference.semconv.trace import SpanAttributes

    ctx = get_current()
    ctx = set_value(SpanAttributes.SESSION_ID, _session_id, ctx)
    ctx = set_value(SpanAttributes.USER_ID, _user_id, ctx)
    attach(ctx)


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
        from droidrun.telemetry.langfuse_processor import (
            get_root_span_context,
            get_last_step_span_context,
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
