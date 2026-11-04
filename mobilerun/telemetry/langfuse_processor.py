"""OpenTelemetry span preprocessing for the Langfuse integration.

Langfuse owns span export, batching, and media upload. Its processor receives
a normalized export snapshot; other processors retain the original ended span.
"""

import base64
import json
import logging
from contextvars import ContextVar
from copy import copy
from typing import TYPE_CHECKING, Any, Optional

from opentelemetry import trace
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from mobilerun import __version__

if TYPE_CHECKING:
    from mobilerun import MobileAgent

_current_agent: ContextVar[Optional["MobileAgent"]] = ContextVar(
    "_current_agent", default=None
)
_root_span_context: ContextVar[Optional[Context]] = ContextVar(
    "_root_span_context", default=None
)
_last_step_span_context: ContextVar[Optional[Context]] = ContextVar(
    "_last_step_span_context", default=None
)

MAX_IMAGE_SIZE_KB = 10000

logger = logging.getLogger("mobilerun")


def set_current_agent(agent: "MobileAgent") -> None:
    _current_agent.set(agent)


def set_root_span_context(span: Span) -> None:
    """Store the root span context for screenshots created outside an active span."""
    try:
        _root_span_context.set(trace.set_span_in_context(span))
    except Exception:
        pass


def get_root_span_context() -> Optional[Context]:
    return _root_span_context.get()


def set_last_step_span_context(span: Span) -> None:
    try:
        _last_step_span_context.set(trace.set_span_in_context(span))
    except Exception:
        pass


def get_last_step_span_context() -> Optional[Context]:
    return _last_step_span_context.get()


class _LangfuseSpanProcessor(SpanProcessor):
    """Wrap the SDK processor without adding an exporter, queue, or uploader."""

    def __init__(self, processor: SpanProcessor, normalizer: "LangfuseSpanProcessor"):
        self.processor = processor
        self.normalizer = normalizer

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        self.normalizer.on_start(span, parent_context)
        self.processor.on_start(span, parent_context)

    def _on_ending(self, span: Span) -> None:
        on_ending = getattr(self.processor, "_on_ending", None)
        if on_ending is not None:
            on_ending(span)

    def on_end(self, span: ReadableSpan) -> None:
        # Ended spans can already be frozen before _on_ending runs. A shallow
        # copy retains all span metadata, including bounded events/links and
        # their dropped counts; only attributes need independent writable storage.
        snapshot = copy(span)
        snapshot._attributes = BoundedAttributes(
            attributes=dict(span.attributes or {}),
            immutable=False,
            extended_attributes=getattr(
                span._attributes, "_extended_attributes", False
            ),
        )
        snapshot._attributes.dropped = span.dropped_attributes
        self.normalizer.on_end(snapshot)
        snapshot._attributes._immutable = True
        self.processor.on_end(snapshot)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.processor.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self.processor.shutdown()


class _LangfuseTracerProvider:
    """Let the public Langfuse client register its processor on our provider."""

    def __init__(self, provider, normalizer: "LangfuseSpanProcessor"):
        self._provider = provider
        self._normalizer = normalizer

    def add_span_processor(self, processor: SpanProcessor) -> None:
        self._provider.add_span_processor(
            _LangfuseSpanProcessor(processor, self._normalizer)
        )

    def __getattr__(self, name):
        return getattr(self._provider, name)


class LangfuseSpanProcessor(SpanProcessor):
    """Enrich live spans and normalize writable Langfuse export snapshots.

    This processor deliberately does not export, batch, upload, or own threads.
    Use it through _LangfuseTracerProvider so on_end only receives private copies.
    """

    def __init__(self, agent: Optional["MobileAgent"] = None) -> None:
        if agent is not None:
            set_current_agent(agent)

    @property
    def agent(self) -> Optional["MobileAgent"]:
        return _current_agent.get()

    def _extract_agent_input(self) -> Optional[dict[str, Any]]:
        """Collect run settings and active LLM metadata for the root observation.

        Action selectors may expose multiple native LLMs, or none for non-LLM
        decision models. Keep planning/helper roles separate from action models.
        """
        agent = self.agent
        if agent is None:
            return None

        try:
            input_data: dict[str, Any] = {}

            if agent.shared_state.instruction:
                input_data["goal"] = agent.shared_state.instruction

            input_data["reasoning"] = agent.config.agent.reasoning

            if agent.config.device:
                device = agent.config.device
                input_data["device"] = {
                    "platform": device.platform,
                    "serial": device.serial,
                    "use_tcp": device.use_tcp,
                }

            if agent.output_model:
                input_data["output_model"] = agent.output_model.__name__

            input_data["droidrun_version"] = "v" + __version__

            if agent.config.agent.after_sleep_action:
                input_data["after_action_sleep"] = agent.config.agent.after_sleep_action

            vision_state = {
                "manager": getattr(agent.config.agent.manager, "vision", False),
                "executor": getattr(agent.config.agent.executor, "vision", False),
                "fast_agent": getattr(agent.config.agent.fast_agent, "vision", False),
            }
            input_data["vision_enabled"] = any(vision_state.values())
            input_data["vision"] = vision_state

            # The selector owns action selection now. Legacy executor_llm and
            # fast_agent_llm attributes may not describe the active model(s).
            action_role = "executor" if agent.config.agent.reasoning else "fast_agent"
            role_llms = []
            if agent.config.agent.reasoning:
                role_llms.append(("manager", agent.manager_llm))
            action_selector = agent.action_selector
            if action_selector is not None:
                role_llms.extend(
                    (action_role, llm) for llm in action_selector.llms
                )
            role_llms.append(("app_opener", agent.app_opener_llm))
            if agent.output_model:
                role_llms.append(("structured_output", agent.structured_output_llm))

            active_llms = []
            for role, llm in role_llms:
                if llm is None:
                    continue

                llm_info: dict[str, Any] = {
                    "role": role,
                    "provider": (
                        llm.class_name() if hasattr(llm, "class_name") else "unknown"
                    ),
                }
                if role in vision_state:
                    llm_info["vision"] = vision_state[role]

                if hasattr(llm, "model"):
                    llm_info["model"] = llm.model
                elif hasattr(llm, "metadata") and hasattr(llm.metadata, "model_name"):
                    llm_info["model"] = llm.metadata.model_name

                if hasattr(llm, "temperature"):
                    llm_info["temperature"] = llm.temperature

                active_llms.append(llm_info)

            input_data["llms"] = active_llms
            return input_data
        except Exception as error:
            logger.warning(
                "Failed to extract Langfuse agent metadata (%s)",
                type(error).__name__,
            )
            return None

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        del parent_context

        agent = self.agent
        attrs = getattr(span, "_attributes", None)
        if attrs is None or agent is None:
            return

        try:
            attrs.pop("input.value", None)

            if span.name == "MobileAgent.run":
                set_root_span_context(span)
                attrs["langfuse.release"] = "v" + __version__
                input_data = self._extract_agent_input()
                if input_data:
                    attrs["langfuse.observation.input"] = json.dumps(input_data)
                    attrs["langfuse.trace.tags"] = (
                        ["reasoning"] if input_data.get("reasoning") else ["fast"]
                    )
                    attrs["droidrun.vision.enabled"] = input_data["vision_enabled"]

            elif span.name in (
                "ManagerAgent.run",
                "StatelessManagerAgent.run",
                "FastAgent.run",
                "ExecutorAgent.run",
            ):
                set_last_step_span_context(span)
                memory = agent.shared_state.agent_memory
                # Step counts/subgoals have a narrower schema than root metadata.
                step_input: dict[str, int | str] = {
                    "memory_size": len(memory) if memory else 0,
                    "message_history_count": len(agent.shared_state.message_history)
                    + 1,
                }

                if span.name == "ExecutorAgent.run":
                    step_input["subgoal"] = (
                        agent.shared_state.current_subgoal or "Unknown"
                    )

                attrs["langfuse.observation.input"] = json.dumps(step_input)
                if agent.shared_state.error_flag_plan:
                    attrs["langfuse.trace.tags"] = ["error_recovery"]
        except Exception as error:
            logger.warning(
                "Failed to add Langfuse span metadata (%s)", type(error).__name__
            )

    def on_end(self, span: ReadableSpan) -> None:
        attrs = getattr(span, "_attributes", None)
        if attrs is None:
            return

        try:
            if span.name.endswith("_done"):
                attrs["langfuse.observation.level"] = "DEBUG"

            if span.name in (
                "MobileAgent.run",
                "ManagerAgent.run",
                "StatelessManagerAgent.run",
                "ExecutorAgent.run",
                "FastAgent.run",
            ):
                attrs.pop("input.value", None)
            elif span.name.endswith(
                (".chat", ".achat", ".stream_chat", ".astream_chat")
            ):
                self._format_chat(span)
            elif span.name.endswith(
                (".complete", ".acomplete", ".stream_complete", ".astream_complete")
            ):
                self._format_complete(span)
            elif span.name == "droidrun.screenshot":
                self._process_screenshot_span(span)

            output = attrs.pop("output.value", None)
            if output is not None and not attrs.get("langfuse.observation.output"):
                attrs["langfuse.observation.output"] = output
        except Exception as error:
            logger.warning(
                "Failed to preprocess a span for Langfuse (%s)",
                type(error).__name__,
            )

    def shutdown(self) -> None:
        """No-op: the public Langfuse client owns exporter shutdown."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """The processor has no queue; all preprocessing is synchronous."""
        del timeout_millis
        return True

    def _format_complete(self, span: ReadableSpan) -> None:
        attrs = getattr(span, "_attributes", None)
        if attrs is None:
            return

        prompts = attrs.pop("llm.prompts", None)
        if isinstance(prompts, (list, tuple)) and prompts:
            attrs["input.value"] = prompts[0]
        self._process_field(attrs, "input")
        self._process_field(attrs, "output")

    def _format_chat(self, span: ReadableSpan) -> None:
        attrs = getattr(span, "_attributes", None)
        if attrs is None:
            return

        self._process_field(attrs, "input")
        self._process_field(attrs, "output")

    def _process_field(self, attrs: dict, field: str) -> None:
        """Normalize an OpenInference input/output value for Langfuse."""
        field_key = f"{field}.value"
        value = attrs.get(field_key)
        if not isinstance(value, str):
            return

        try:
            data = json.loads(value)
            if self._has_blocks_to_transform(data):
                self._transform_and_set_field(attrs, field, data)
                return
            sanitized, contains_data_uri = self._sanitize_serialized_data_uris(data)
            if contains_data_uri:
                attrs[f"langfuse.observation.{field}"] = json.dumps(sanitized)
                attrs.pop(field_key, None)
                return
        except (json.JSONDecodeError, ValueError):
            pass

        attrs[f"langfuse.observation.{field}"] = value
        attrs.pop(field_key, None)

    @staticmethod
    def _has_blocks_to_transform(data: object) -> bool:
        if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
            return False

        return any(
            isinstance(message, dict)
            and (
                "blocks" in message
                or (
                    isinstance(message.get("json"), dict)
                    and "blocks" in message["json"]
                )
            )
            for message in data["messages"]
        )

    def _transform_and_set_field(self, attrs: dict, field: str, data: dict) -> None:
        prefix = f"llm.{field}_messages."
        for key in [key for key in attrs if key.startswith(prefix)]:
            del attrs[key]

        formatted = self._transform_blocks_to_content(data)
        attrs[f"langfuse.observation.{field}"] = formatted
        attrs.pop(f"{field}.value", None)

    def _transform_blocks_to_content(self, data: dict) -> str:
        processed = self._convert_message_array(data["messages"])
        return json.dumps({"messages": processed})

    def _convert_message_array(self, messages: list) -> list:
        restructured_messages = []

        for original_message in messages:
            if not isinstance(original_message, dict):
                continue

            message = original_message
            if "content" in message and "blocks" not in message:
                restructured_messages.append(message)
                continue

            if isinstance(message.get("json"), dict) and "blocks" in message["json"]:
                message = message.copy()
                message.update(message.pop("json"))

            if "blocks" not in message or "role" not in message:
                if "role" in message:
                    restructured_messages.append(message)
                continue

            blocks = message["blocks"]
            if not isinstance(blocks, list) or not blocks:
                restructured_messages.append(message)
                continue

            role = message["role"]
            if (
                len(blocks) == 1
                and isinstance(blocks[0], dict)
                and blocks[0].get("block_type") == "text"
                and "text" in blocks[0]
            ):
                restructured_messages.append(
                    {"role": role, "content": blocks[0]["text"]}
                )
                continue

            content_blocks = self._convert_blocks_to_content(blocks)
            if content_blocks:
                restructured_messages.append({"role": role, "content": content_blocks})
            elif any(
                isinstance(block, dict) and block.get("block_type") == "image"
                for block in blocks
            ):
                # Never restore rejected image bytes into exported attributes.
                restructured_messages.append({"role": role, "content": []})
            else:
                restructured_messages.append(message)

        return restructured_messages

    def _convert_blocks_to_content(self, blocks: list) -> list:
        content_blocks = []

        for block in blocks:
            if not isinstance(block, dict):
                continue

            block_type = block.get("block_type")
            if block_type == "text" and "text" in block:
                content_blocks.append({"type": "text", "text": block["text"]})
            elif block_type == "image":
                image_block = self._prepare_image_for_native_upload(block)
                if image_block:
                    content_blocks.append(image_block)
            elif (
                block_type == "tool_call"
                and "tool_name" in block
                and "tool_kwargs" in block
            ):
                content_blocks.append(
                    {
                        "type": "tool_call",
                        "tool_call": {
                            "name": block["tool_name"],
                            "arguments": block["tool_kwargs"],
                        },
                    }
                )

        return content_blocks

    @classmethod
    def _sanitize_serialized_data_uris(cls, value: Any) -> tuple[Any, bool]:
        """Apply media limits to base64 data URIs already serialized as content."""
        if isinstance(value, str):
            header, separator, _payload = value.partition(",")
            is_base64_data_uri = (
                value.startswith("data:")
                and bool(separator)
                and header.endswith(";base64")
            )
            if not is_base64_data_uri:
                return value, False

            prepared = cls._prepare_image_for_native_upload(
                {"image": value, "image_mimetype": "application/octet-stream"}
            )
            return (value if prepared else ""), True

        if isinstance(value, list):
            sanitized_items = []
            contains_data_uri = False
            for item in value:
                sanitized, found = cls._sanitize_serialized_data_uris(item)
                sanitized_items.append(sanitized)
                contains_data_uri = contains_data_uri or found
            return sanitized_items, contains_data_uri

        if isinstance(value, dict):
            sanitized_mapping = {}
            contains_data_uri = False
            for key, item in value.items():
                sanitized, found = cls._sanitize_serialized_data_uris(item)
                sanitized_mapping[key] = sanitized
                contains_data_uri = contains_data_uri or found
            return sanitized_mapping, contains_data_uri

        return value, False

    def _process_screenshot_span(self, span: ReadableSpan) -> None:
        attrs = getattr(span, "_attributes", None)
        if attrs is None:
            return

        image = attrs.get("droidrun.screenshot.image_base64")
        mime_type = attrs.get("droidrun.screenshot.mime_type", "image/png")
        if not image:
            return

        image_content = self._prepare_image_for_native_upload(
            {
                "image": image,
                "image_mimetype": mime_type,
            }
        )
        if image_content:
            attrs["langfuse.observation.output"] = json.dumps(
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [image_content],
                        }
                    ]
                }
            )

        attrs.pop("droidrun.screenshot.image_base64", None)
        attrs.pop("droidrun.screenshot.mime_type", None)
        attrs.pop("output.value", None)

    @staticmethod
    def _prepare_image_for_native_upload(block: dict) -> Optional[dict]:
        """Return media in a form Langfuse v4 uploads during span export."""
        image = block.get("image")
        if image is not None:
            mime_type = block.get("image_mimetype")
            if not isinstance(image, str):
                logger.warning("Image data is invalid; skipping upload")
                return None

            encoded = image
            if image.startswith("data:") and "," in image:
                header, encoded = image.split(",", 1)
                if header.endswith(";base64"):
                    mime_type = header[5:-7]
                else:
                    logger.warning(
                        "Image data URI is not base64 encoded; skipping upload"
                    )
                    return None
            elif not isinstance(mime_type, str):
                logger.warning("Image MIME type is invalid; skipping upload")
                return None

            try:
                image_bytes = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError):
                logger.warning("Image data is not valid base64; skipping upload")
                return None

            size_kb = len(image_bytes) / 1024
            if size_kb > MAX_IMAGE_SIZE_KB:
                logger.warning(
                    "Image size (%.1fKB) exceeds limit (%dKB); skipping upload",
                    size_kb,
                    MAX_IMAGE_SIZE_KB,
                )
                return None

            data_uri = f"data:{mime_type};base64,{encoded}"
            return {"type": "image_url", "image_url": {"url": data_uri}}

        url = block.get("url")
        if isinstance(url, str):
            if url.startswith("data:"):
                return LangfuseSpanProcessor._prepare_image_for_native_upload(
                    {
                        "image": url,
                        "image_mimetype": "application/octet-stream",
                    }
                )
            return {"type": "image_url", "image_url": {"url": url}}

        path = block.get("path")
        if isinstance(path, str):
            logger.warning("Using a local image path; it may not resolve in Langfuse")
            return {"type": "image_url", "image_url": {"url": f"file://{path}"}}

        return None
