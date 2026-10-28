import asyncio
import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from opentelemetry import trace
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import (
    ReadableSpan,
    SpanLimits,
    SpanProcessor,
    TracerProvider,
)
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from selector_helpers import StubSelector

from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.utils import tracing_setup
from mobilerun.config_manager.config_manager import TracingConfig
from mobilerun.telemetry import langfuse_processor
from mobilerun.telemetry.langfuse_processor import (
    LangfuseSpanProcessor,
    _LangfuseSpanProcessor,
    _LangfuseTracerProvider,
)


@pytest.fixture(autouse=True)
def reset_tracing_state(monkeypatch):
    monkeypatch.setattr(tracing_setup, "_tracing_initialized", False)
    monkeypatch.setattr(tracing_setup, "_tracing_provider", None)
    monkeypatch.setattr(
        tracing_setup,
        "_legacy_session_context",
        (tracing_setup._default_session_id, "anonymous"),
    )
    monkeypatch.setattr(tracing_setup, "_langfuse_client", None)
    monkeypatch.setattr(tracing_setup, "_langfuse_preprocessor", None)
    monkeypatch.setattr(tracing_setup, "_langfuse_tracer_provider", None)
    monkeypatch.setattr(tracing_setup, "_langfuse_setup_attempted", False)
    for name in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_HOST",
    ):
        monkeypatch.delenv(name, raising=False)

    agent_token = langfuse_processor._current_agent.set(None)
    root_token = langfuse_processor._root_span_context.set(None)
    step_token = langfuse_processor._last_step_span_context.set(None)
    yield
    langfuse_processor._current_agent.reset(agent_token)
    langfuse_processor._root_span_context.reset(root_token)
    langfuse_processor._last_step_span_context.reset(step_token)


class _FakeTracerProvider:
    def __init__(self):
        self.processors = []

    def add_span_processor(self, processor):
        self.processors.append(processor)


def _install_fake_setup(
    monkeypatch, *, auth_result=True, auth_error=None, attach_exporter=True
):
    import langfuse
    from openinference.instrumentation import llama_index as instrumentation
    from openinference.instrumentation.llama_index import _handler

    provider = _FakeTracerProvider()
    clients = []

    class FakeExporter:
        pass

    FakeExporter.__module__ = "langfuse._client.span_processor"
    exporter = FakeExporter()
    instrument_calls = []

    class FakeInstrumentor:
        is_instrumented_by_opentelemetry = False

        def instrument(self):
            instrument_calls.append(True)

    class FakeLangfuse:
        def __init__(self, **kwargs):
            time.sleep(0.01)
            self.kwargs = kwargs
            clients.append(self)
            if attach_exporter:
                kwargs["tracer_provider"].add_span_processor(exporter)

        def auth_check(self):
            if auth_error:
                raise auth_error
            return auth_result

    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(instrumentation, "LlamaIndexInstrumentor", FakeInstrumentor)
    monkeypatch.setattr(_handler, "_encoder", lambda obj: obj)
    monkeypatch.setattr(langfuse, "Langfuse", FakeLangfuse)
    return provider, clients, exporter, instrument_calls


def _langfuse_config(**kwargs):
    values = {
        "enabled": True,
        "provider": "langfuse",
        "langfuse_public_key": "pk-test",
        "langfuse_secret_key": "sk-test",
    }
    values.update(kwargs)
    return TracingConfig(**values)


def test_setup_is_concurrent_safe_and_wraps_one_sdk_processor(monkeypatch):
    provider, clients, exporter, instrument_calls = _install_fake_setup(monkeypatch)
    config = _langfuse_config()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: tracing_setup.setup_tracing(config), range(16)))

    assert len(clients) == 1
    assert instrument_calls == [True]
    assert len(provider.processors) == 1
    assert isinstance(provider.processors[0], _LangfuseSpanProcessor)
    assert provider.processors[0].processor is exporter
    assert provider.processors[0].normalizer is tracing_setup._langfuse_preprocessor
    assert clients[0].kwargs["should_export_span"](object()) is True
    assert tracing_setup._tracing_initialized is True
    assert tracing_setup._tracing_provider == "langfuse"


@pytest.mark.parametrize(
    ("auth_result", "auth_error"),
    [(False, None), (True, RuntimeError("sensitive-detail"))],
)
def test_auth_diagnostic_does_not_retry_or_leak_details(
    monkeypatch, caplog, auth_result, auth_error
):
    _provider, clients, _exporter, _calls = _install_fake_setup(
        monkeypatch, auth_result=auth_result, auth_error=auth_error
    )

    tracing_setup.setup_tracing(_langfuse_config())
    tracing_setup.setup_tracing(_langfuse_config())

    assert len(clients) == 1
    assert tracing_setup._tracing_initialized is True
    assert "sensitive-detail" not in caplog.text


@pytest.mark.parametrize("register_before_failure", [False, True])
def test_failed_client_construction_cannot_accumulate_processors(
    monkeypatch, register_before_failure
):
    import langfuse
    from openinference.instrumentation import llama_index as instrumentation
    from openinference.instrumentation.llama_index import _handler

    provider = _FakeTracerProvider()
    construction_count = 0

    class FakeInstrumentor:
        is_instrumented_by_opentelemetry = True

    class FailingLangfuse:
        def __init__(self, **_kwargs):
            nonlocal construction_count
            construction_count += 1
            if register_before_failure:
                _kwargs["tracer_provider"].add_span_processor(SpanProcessor())
            raise ValueError("sensitive-detail")

    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(instrumentation, "LlamaIndexInstrumentor", FakeInstrumentor)
    monkeypatch.setattr(_handler, "_encoder", lambda obj: obj)
    monkeypatch.setattr(langfuse, "Langfuse", FailingLangfuse)

    tracing_setup.setup_tracing(_langfuse_config())
    tracing_setup.setup_tracing(_langfuse_config())

    assert construction_count == 1
    assert len(provider.processors) == int(register_before_failure)
    if register_before_failure:
        assert isinstance(provider.processors[0], _LangfuseSpanProcessor)
    assert tracing_setup._tracing_initialized is False


@pytest.mark.parametrize("wrapped", [False, True])
def test_setup_rejects_preexisting_langfuse_exporter(monkeypatch, caplog, wrapped):
    provider, clients, _exporter, _calls = _install_fake_setup(monkeypatch)

    class ExistingLangfuseExporter:
        pass

    ExistingLangfuseExporter.__module__ = "langfuse._client.span_processor"
    processor = ExistingLangfuseExporter()
    if wrapped:
        processor = _LangfuseSpanProcessor(processor, LangfuseSpanProcessor())
    provider._active_span_processor = SimpleNamespace(_span_processors=(processor,))

    tracing_setup.setup_tracing(_langfuse_config())

    assert clients == []
    assert provider.processors == []
    assert tracing_setup._tracing_initialized is False
    assert "pk-test" not in caplog.text
    assert "sk-test" not in caplog.text


def test_setup_rejects_same_key_client_owned_by_another_provider(monkeypatch, caplog):
    provider, clients, _exporter, _calls = _install_fake_setup(
        monkeypatch, attach_exporter=False
    )

    tracing_setup.setup_tracing(_langfuse_config())

    assert len(clients) == 1
    assert provider.processors == []
    assert tracing_setup._langfuse_client is None
    assert tracing_setup._tracing_initialized is False
    assert "pk-test" not in caplog.text
    assert "sk-test" not in caplog.text


@pytest.mark.parametrize(
    ("explicit", "base_env", "legacy_env", "expected"),
    [
        ("https://explicit", "https://base", "https://legacy", "https://explicit"),
        ("", "https://base", "https://legacy", "https://base"),
        ("", "", "https://legacy", "https://legacy"),
        ("", "", "", tracing_setup.DEFAULT_LANGFUSE_BASE_URL),
    ],
)
def test_langfuse_base_url_precedence(
    monkeypatch, explicit, base_env, legacy_env, expected
):
    if base_env:
        monkeypatch.setenv("LANGFUSE_BASE_URL", base_env)
    if legacy_env:
        monkeypatch.setenv("LANGFUSE_HOST", legacy_env)

    config = _langfuse_config(langfuse_host=explicit)
    assert tracing_setup._resolve_langfuse_base_url(config) == expected


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("selector_kind", ["none", "non_llm", "single_llm", "multi_llm"])
def test_processor_normalizes_agent_and_llm_metadata(reasoning, selector_kind):
    class LLM:
        model = "test-model"
        temperature = 0.2

        @staticmethod
        def class_name():
            return "TestLLM"

    selector_llms = (LLM(),) * {"single_llm": 1, "multi_llm": 2}.get(selector_kind, 0)
    if selector_kind == "none":
        selector = None
    elif selector_kind == "non_llm":
        # Non-LLM selectors explicitly expose an empty native-LLM collection.
        selector = StubSelector(llms=())
    else:
        selector = StubSelector(llms=selector_llms)

    agent = SimpleNamespace(
        shared_state=SimpleNamespace(
            instruction="Open Settings",
            agent_memory="memory",
            message_history=[],
            current_subgoal=None,
            error_flag_plan=False,
        ),
        config=SimpleNamespace(
            agent=SimpleNamespace(
                reasoning=reasoning,
                after_sleep_action=1,
                manager=SimpleNamespace(vision=False),
                executor=SimpleNamespace(vision=False),
                fast_agent=SimpleNamespace(vision=True),
            ),
            device=SimpleNamespace(
                platform="android", serial="emulator", use_tcp=False
            ),
        ),
        output_model=SimpleNamespace(__name__="Output"),
        action_selector=selector,
        manager_llm=LLM() if reasoning else None,
        app_opener_llm=LLM(),
        structured_output_llm=LLM(),
        # These compatibility attributes must not override selector metadata.
        executor_llm=SimpleNamespace(model="unused-executor"),
        fast_agent_llm=SimpleNamespace(model="unused-fast-agent"),
    )
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    _LangfuseTracerProvider(provider, LangfuseSpanProcessor(agent)).add_span_processor(
        SimpleSpanProcessor(exporter)
    )

    with provider.get_tracer("test").start_as_current_span("MobileAgent.run"):
        pass

    attrs = exporter.get_finished_spans()[0].attributes
    metadata = json.loads(attrs["langfuse.observation.input"])
    assert metadata["goal"] == "Open Settings"
    assert metadata["vision_enabled"] is True
    assert metadata["output_model"] == "Output"
    assert metadata["reasoning"] is reasoning
    action_role = "executor" if reasoning else "fast_agent"
    expected_roles = (["manager"] if reasoning else []) + [action_role] * len(selector_llms)
    expected_roles.extend(["app_opener", "structured_output"])
    expected_llms = []
    for role in expected_roles:
        info = {
            "role": role,
            "provider": "TestLLM",
            "model": "test-model",
            "temperature": 0.2,
        }
        if role in ("manager", "executor", "fast_agent"):
            info["vision"] = role == "fast_agent"
        expected_llms.append(info)
    assert metadata["llms"] == expected_llms
    assert attrs["langfuse.trace.tags"] == (("reasoning",) if reasoning else ("fast",))
    provider.shutdown()


@pytest.mark.parametrize(
    "span_name",
    [
        "MobileAgent.run",
        "ManagerAgent.run",
        "StatelessManagerAgent.run",
        "FastAgent.run",
        "ExecutorAgent.run",
    ],
)
def test_processor_skips_agent_metadata_without_current_agent(span_name):
    processor = LangfuseSpanProcessor()
    span = Mock()
    span.name = span_name
    span._attributes = {"input.value": "original"}

    processor.on_start(span)

    assert processor._extract_agent_input() is None
    assert span._attributes == {"input.value": "original"}


@pytest.mark.parametrize(
    "span_name",
    [
        "ManagerAgent.run",
        "StatelessManagerAgent.run",
        "FastAgent.run",
        "ExecutorAgent.run",
    ],
)
@pytest.mark.parametrize("subgoal", ["", "Tap Settings"])
def test_processor_adds_step_metadata(span_name, subgoal):
    agent = Mock()
    agent.shared_state = MobileAgentState(
        agent_memory="memory",
        current_subgoal=subgoal,
        error_flag_plan=True,
    )
    processor = LangfuseSpanProcessor(agent)
    span = Mock()
    span.name = span_name
    span._attributes = {}

    processor.on_start(span)

    expected: dict[str, int | str] = {
        "memory_size": len("memory"),
        "message_history_count": 1,
    }
    if span_name == "ExecutorAgent.run":
        expected["subgoal"] = subgoal or "Unknown"
    assert json.loads(span._attributes["langfuse.observation.input"]) == expected
    assert span._attributes["langfuse.trace.tags"] == ["error_recovery"]


class _CollectingExporter(SpanExporter):
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


@pytest.mark.parametrize(
    "span_name", ["droidrun.screenshot", "LLM.achat", "LLM.acomplete"]
)
def test_public_v4_client_exports_custom_span_once_and_uploads_native_media(
    monkeypatch,
    span_name,
):
    from langfuse import Langfuse
    from langfuse._task_manager.media_manager import MediaManager

    media_jobs = []

    def record_media(_self, *, data):
        media_jobs.append(data)

    monkeypatch.setattr(MediaManager, "_process_upload_media_job", record_media)
    provider = TracerProvider()
    exporter = _CollectingExporter()
    preprocessor = LangfuseSpanProcessor()
    before, after = InMemorySpanExporter(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(before))
    client = Langfuse(
        public_key=f"pk-test-{uuid4()}",
        secret_key="sk-test",
        base_url="http://127.0.0.1:1",
        tracer_provider=_LangfuseTracerProvider(provider, preprocessor),
        span_exporter=exporter,
        should_export_span=lambda _span: True,
    )
    provider.add_span_processor(SimpleSpanProcessor(after))

    try:
        image = base64.b64encode(b"png-bytes").decode()
        if span_name == "droidrun.screenshot":
            original_attrs = {
                "droidrun.screenshot.image_base64": image,
                "droidrun.screenshot.mime_type": "image/png",
            }
            media_field = "langfuse.observation.output"
        else:
            messages = json.dumps(
                {
                    "messages": [
                        {
                            "role": "user",
                            "blocks": [
                                {
                                    "block_type": "image",
                                    "image": image,
                                    "image_mimetype": "image/png",
                                }
                            ],
                        }
                    ]
                }
            )
            original_attrs = (
                {"llm.prompts": (messages,)}
                if span_name.endswith("complete")
                else {"input.value": messages}
            )
            media_field = "langfuse.observation.input"
        with provider.get_tracer("mobilerun.custom").start_as_current_span(
            span_name, attributes=original_attrs
        ) as span:
            started_attrs = dict(span.attributes)
        client.flush()
        assert provider.force_flush() is True

        assert len(exporter.spans) == 1
        attrs = exporter.spans[0].attributes
        assert "droidrun.screenshot.image_base64" not in attrs
        assert "@@@langfuseMedia:type=image/png" in attrs[media_field]
        assert image not in json.dumps(dict(attrs))
        assert len(media_jobs) == 1
        assert media_jobs[0]["content_bytes"] == b"png-bytes"
        assert media_jobs[0]["media_id"] in attrs[media_field]
        assert media_jobs[0]["observation_id"] == format(
            exporter.spans[0].context.span_id, "016x"
        )
        original = before.get_finished_spans()[0]
        assert after.get_finished_spans()[0] is original
        assert original.attributes == started_attrs
        assert media_field not in original.attributes
        assert exporter.spans[0].context == original.context
    finally:
        client.shutdown()
        provider.shutdown()


@pytest.mark.parametrize(
    ("name", "attributes", "expected_input", "expected_output"),
    [
        (
            "LLM.achat",
            {
                "input.value": '{"messages": [{"role": "user", "blocks": [{"block_type": "text", "text": "hi"}]}]}',
                "output.value": "hello",
            },
            {"messages": [{"role": "user", "content": "hi"}]},
            "hello",
        ),
        (
            "LLM.astream_complete",
            {"llm.prompts": ("hello",), "output.value": "world"},
            "hello",
            "world",
        ),
        (
            "MobileAgent.run",
            {"input.value": "internal", "output.value": "done"},
            None,
            "done",
        ),
        ("step_done", {}, None, None),
    ],
)
def test_frozen_span_normalization(name, attributes, expected_input, expected_output):
    original = ReadableSpan(name=name, attributes=MappingProxyType(attributes))
    processor = Mock(spec=SpanProcessor)
    _LangfuseSpanProcessor(processor, LangfuseSpanProcessor()).on_end(original)
    processor.on_end.assert_called_once()
    snapshot = processor.on_end.call_args.args[0]
    assert snapshot is not original
    assert dict(original.attributes) == attributes
    attrs = snapshot.attributes
    actual_input = attrs.get("langfuse.observation.input")
    if isinstance(expected_input, dict):
        actual_input = json.loads(actual_input)
    assert actual_input == expected_input
    assert attrs.get("langfuse.observation.output") == expected_output
    assert "input.value" not in attrs
    assert "output.value" not in attrs
    if name.endswith("_done"):
        assert attrs["langfuse.observation.level"] == "DEBUG"


def test_snapshot_preserves_span_metadata_and_dropped_counts():
    provider = TracerProvider(
        resource=Resource.create({"service.name": "snapshot-test"}),
        span_limits=SpanLimits(max_attributes=2, max_events=1, max_links=1),
    )
    original_exporter, normalized_exporter = (
        InMemorySpanExporter(),
        InMemorySpanExporter(),
    )
    provider.add_span_processor(SimpleSpanProcessor(original_exporter))
    _LangfuseTracerProvider(provider, LangfuseSpanProcessor()).add_span_processor(
        SimpleSpanProcessor(normalized_exporter)
    )
    tracer = provider.get_tracer("test-library", "1.2.3", "https://schema.example")
    try:
        with tracer.start_as_current_span("parent") as parent:
            links = [trace.Link(parent.get_span_context()) for _ in range(2)]
            with tracer.start_as_current_span("child", links=links) as span:
                span.set_attributes(
                    {"dropped": "yes", "kept": "yes", "output.value": "result"}
                )
                span.add_event("dropped-event")
                span.add_event("kept-event", {"event-key": "value"})
                span.set_status(trace.Status(trace.StatusCode.ERROR, "test-status"))
        original = original_exporter.get_finished_spans()[0]
        snapshot = normalized_exporter.get_finished_spans()[0]
        for field in (
            "name",
            "context",
            "parent",
            "resource",
            "kind",
            "start_time",
            "end_time",
            "status",
            "events",
            "links",
            "instrumentation_scope",
            "dropped_attributes",
            "dropped_events",
            "dropped_links",
        ):
            assert getattr(snapshot, field) == getattr(original, field), field
        with pytest.warns(DeprecationWarning, match="instrumentation_scope"):
            assert snapshot.instrumentation_info == original.instrumentation_info
        assert (
            snapshot.dropped_attributes,
            snapshot.dropped_events,
            snapshot.dropped_links,
        ) == (1, 1, 1)
        assert original.attributes == {"kept": "yes", "output.value": "result"}
        assert snapshot.attributes == {
            "kept": "yes",
            "langfuse.observation.output": "result",
        }
    finally:
        provider.shutdown()


def test_snapshot_preserves_extended_attributes():
    attributes = BoundedAttributes(
        attributes={"custom": {"nested": ["value"]}, "output.value": "done"},
        immutable=True,
        extended_attributes=True,
    )
    original = ReadableSpan(name="span", attributes=attributes)
    processor = Mock(spec=SpanProcessor)
    _LangfuseSpanProcessor(processor, LangfuseSpanProcessor()).on_end(original)
    snapshot = processor.on_end.call_args.args[0]
    assert snapshot.attributes["custom"] == original.attributes["custom"]
    assert original.attributes["output.value"] == "done"
    assert snapshot.attributes["langfuse.observation.output"] == "done"


def test_adapter_delegates_tracing_and_lifecycle():
    provider = TracerProvider()
    adapter = _LangfuseTracerProvider(provider, LangfuseSpanProcessor())
    processor = Mock(spec=SpanProcessor)
    processor.force_flush.return_value = False
    adapter.add_span_processor(processor)
    with adapter.get_tracer("test").start_as_current_span("span") as span:
        pass
    processor.on_start.assert_called_once_with(span, None)
    processor._on_ending.assert_called_once_with(span)
    processor.on_end.assert_called_once()
    assert adapter.force_flush(123) is False
    assert processor.force_flush.call_count == 1
    # TracerProvider subtracts elapsed time from its flush deadline.
    assert 0 <= processor.force_flush.call_args.args[0] <= 123
    adapter.shutdown()
    processor.shutdown.assert_called_once()


def test_owned_pipeline_rejects_duplicate_or_foreign_normalizers(monkeypatch):
    provider, _clients, exporter, _calls = _install_fake_setup(monkeypatch)
    tracing_setup.setup_tracing(_langfuse_config())
    assert tracing_setup._provider_has_owned_langfuse_pipeline(provider)
    owned = provider.processors[0]
    provider.processors.append(exporter)
    assert not tracing_setup._provider_has_owned_langfuse_pipeline(provider)
    provider.processors[:] = [_LangfuseSpanProcessor(exporter, LangfuseSpanProcessor())]
    assert not tracing_setup._provider_has_owned_langfuse_pipeline(provider)
    provider.processors[:] = [owned, owned]
    assert not tracing_setup._provider_has_owned_langfuse_pipeline(provider)


def test_native_image_size_and_error_handling(monkeypatch, caplog):
    monkeypatch.setattr(langfuse_processor, "MAX_IMAGE_SIZE_KB", 0)
    encoded = base64.b64encode(b"x").decode()

    assert (
        LangfuseSpanProcessor._prepare_image_for_native_upload(
            {"image": encoded, "image_mimetype": "image/png"}
        )
        is None
    )
    assert (
        LangfuseSpanProcessor._prepare_image_for_native_upload(
            {"image": "not-base64", "image_mimetype": "image/png"}
        )
        is None
    )
    assert (
        LangfuseSpanProcessor._prepare_image_for_native_upload(
            {"url": f"data:image/png;base64,{encoded}"}
        )
        is None
    )
    assert "not-base64" not in caplog.text


@pytest.mark.parametrize("nested", [False, True])
def test_rejected_image_bytes_are_not_restored_to_observation_attributes(
    monkeypatch, nested
):
    monkeypatch.setattr(langfuse_processor, "MAX_IMAGE_SIZE_KB", 0)
    encoded = base64.b64encode(b"oversized-image").decode()
    message = {
        "role": "user",
        "blocks": [
            {
                "block_type": "image",
                "image": encoded,
                "image_mimetype": "image/png",
            }
        ],
    }
    if nested:
        message = {"json": message}
    attrs = {"input.value": json.dumps({"messages": [message]})}

    LangfuseSpanProcessor()._process_field(attrs, "input")

    assert encoded not in attrs["langfuse.observation.input"]
    assert "input.value" not in attrs


def test_rejected_serialized_content_media_is_removed(monkeypatch):
    monkeypatch.setattr(langfuse_processor, "MAX_IMAGE_SIZE_KB", 0)
    encoded = base64.b64encode(b"oversized-image").decode()
    attrs = {
        "input.value": json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded}"
                                },
                            }
                        ],
                    }
                ]
            }
        )
    }

    LangfuseSpanProcessor()._process_field(attrs, "input")

    assert encoded not in attrs["langfuse.observation.input"]
    assert "input.value" not in attrs


@pytest.mark.parametrize(
    "value",
    [
        "data: ordinary non-media text",
        {"message": "data: nested non-media text"},
        ["data: list-contained non-media text"],
        {"uri": "data:text/plain,hello"},
    ],
)
def test_non_media_data_prefixed_serialized_content_is_preserved(value, caplog):
    serialized = json.dumps(value)
    attrs = {"input.value": serialized}

    LangfuseSpanProcessor()._process_field(attrs, "input")

    assert attrs["langfuse.observation.input"] == serialized
    assert "input.value" not in attrs
    assert "skipping upload" not in caplog.text


def test_mixed_serialized_content_preserves_text_and_removes_oversized_media(
    monkeypatch,
):
    monkeypatch.setattr(langfuse_processor, "MAX_IMAGE_SIZE_KB", 0)
    encoded = base64.b64encode(b"oversized-image").decode()
    attrs = {
        "input.value": json.dumps(
            {
                "content": [
                    "data: ordinary non-media text",
                    f"data:image/png;base64,{encoded}",
                ]
            }
        )
    }

    LangfuseSpanProcessor()._process_field(attrs, "input")

    observed = json.loads(attrs["langfuse.observation.input"])
    assert observed["content"] == ["data: ordinary non-media text", ""]
    assert encoded not in attrs["langfuse.observation.input"]
    assert "input.value" not in attrs


def test_malformed_base64_serialized_media_is_removed(caplog):
    attrs = {
        "input.value": json.dumps({"url": "data:image/png;base64,not-valid-base64"})
    }

    LangfuseSpanProcessor()._process_field(attrs, "input")

    assert json.loads(attrs["langfuse.observation.input"])["url"] == ""
    assert "not-valid-base64" not in caplog.text
    assert "input.value" not in attrs


@pytest.mark.parametrize(
    ("session_id", "user_id", "expected_session", "expected_user"),
    [
        ("session-123", "user-123", "session-123", "user-123"),
        (None, None, tracing_setup._default_session_id, "anonymous"),
    ],
)
def test_session_context_sets_langfuse_trace_attributes(
    monkeypatch, session_id, user_id, expected_session, expected_user
):
    from openinference.semconv.trace import SpanAttributes
    from opentelemetry.context import get_current, get_value

    monkeypatch.setattr(tracing_setup, "_tracing_initialized", True)
    monkeypatch.setattr(tracing_setup, "_tracing_provider", "langfuse")
    before = get_current()
    with tracing_setup.session_context(session_id, user_id):
        assert get_value(SpanAttributes.SESSION_ID) == expected_session
        assert get_value(SpanAttributes.USER_ID) == expected_user
    assert get_current() is before


@pytest.mark.parametrize(
    "session_id,user_id", [("legacy-session", "legacy-user"), (None, None)]
)
def test_legacy_apply_session_context_remains_importable_and_uses_latest_setup(
    monkeypatch, session_id, user_id
):
    from openinference.semconv.trace import SpanAttributes
    from opentelemetry.context import get_value

    from mobilerun.agent.utils.tracing_setup import apply_session_context

    monkeypatch.setattr(tracing_setup, "_setup_langfuse_tracing", lambda *args: True)
    tracing_setup.setup_tracing(
        TracingConfig(
            enabled=True, provider="langfuse", langfuse_session_id="earlier-session"
        )
    )
    tracing_setup.setup_tracing(
        TracingConfig(
            enabled=True,
            provider="langfuse",
            langfuse_session_id=session_id,
            langfuse_user_id=user_id,
        )
    )

    def read_context():
        assert apply_session_context() is None
        return get_value(SpanAttributes.SESSION_ID), get_value(SpanAttributes.USER_ID)

    # Legacy attachment intentionally remains active; isolate it from other tests.
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(read_context).result() == (
            session_id or tracing_setup._default_session_id,
            user_id or "anonymous",
        )


@pytest.mark.parametrize("provider", [None, "phoenix"])
def test_session_context_leaves_other_providers_unchanged(monkeypatch, provider):
    from opentelemetry.context import get_current

    monkeypatch.setattr(tracing_setup, "_tracing_initialized", provider is not None)
    monkeypatch.setattr(tracing_setup, "_tracing_provider", provider)
    before = get_current()
    with tracing_setup.session_context("session", "user"):
        assert get_current() is before
    assert get_current() is before


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_concurrent_mobile_agents_keep_their_tracing_context(monkeypatch, outcome):
    from llama_index.core.workflow import StartEvent, step
    from openinference.instrumentation import TracerProvider as OITracerProvider
    from openinference.semconv.trace import SpanAttributes
    from opentelemetry.context import get_current, get_value
    from workflows.errors import WorkflowCancelledByUser

    from mobilerun.agent.droid.droid_agent import MobileAgent
    from mobilerun.agent.droid.events import ResultEvent
    from mobilerun.config_manager.config_manager import (
        AgentConfig,
        MobileConfig,
        TelemetryConfig,
    )

    # Use the existing process-wide provider path without creating a network client.
    monkeypatch.setattr(tracing_setup, "_tracing_initialized", True)
    monkeypatch.setattr(tracing_setup, "_tracing_provider", "langfuse")
    provider = OITracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    async def exercise():
        barrier = asyncio.Barrier(2)
        entered = asyncio.Event()
        release = asyncio.Event()
        observations = {}

        class ContextAgent(MobileAgent):
            @step
            async def start_handler(self, ev: StartEvent) -> ResultEvent:
                await barrier.wait()
                expected = (self._tracing_session_id, self._tracing_user_id)
                for index in range(3):
                    with tracer.start_as_current_span(
                        f"{self.shared_state.instruction}-{index}"
                    ):
                        await asyncio.sleep(0)
                        actual = (
                            get_value(SpanAttributes.SESSION_ID),
                            get_value(SpanAttributes.USER_ID),
                        )
                        assert actual == expected
                        assert langfuse_processor._current_agent.get() is self
                        assert (
                            await asyncio.to_thread(
                                lambda: (
                                    get_value(SpanAttributes.SESSION_ID),
                                    get_value(SpanAttributes.USER_ID),
                                )
                            )
                            == expected
                        )
                        observations[self.shared_state.instruction] = actual
                if self.shared_state.instruction == "first":
                    entered.set()
                    await release.wait()
                    if outcome == "error":
                        raise RuntimeError("expected run failure")
                return ResultEvent(success=True, reason=expected[0], steps=3)

        agents = [
            ContextAgent(
                name,
                config=MobileConfig(
                    agent=AgentConfig(name="context-test-agent"),
                    telemetry=TelemetryConfig(enabled=False),
                    tracing=TracingConfig(
                        enabled=True,
                        provider="langfuse",
                        langfuse_session_id=f"{name}-session",
                        langfuse_user_id=f"{name}-user",
                    ),
                ),
                disable_validation=True,
            )
            for name in ("first", "second")
        ]
        before = get_current()
        caller_agent = langfuse_processor._current_agent.get()
        first = agents[0].run()
        assert get_current() is before
        assert langfuse_processor._current_agent.get() is caller_agent
        second = agents[1].run()
        assert get_current() is before
        assert langfuse_processor._current_agent.get() is caller_agent
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert (await second).reason == "second-session"
        if outcome == "cancel":
            await first.cancel_run()
            with pytest.raises(WorkflowCancelledByUser):
                await first
        else:
            release.set()
            if outcome == "error":
                with pytest.raises(RuntimeError, match="expected run failure"):
                    await first
            else:
                assert (await first).reason == "first-session"
        assert observations == {
            "first": ("first-session", "first-user"),
            "second": ("second-session", "second-user"),
        }
        assert get_current() is before
        assert langfuse_processor._current_agent.get() is caller_agent

    try:
        asyncio.run(exercise())
        spans = exporter.get_finished_spans()
        assert len(spans) == 6
        for span in spans:
            name = span.name.split("-")[0]
            assert span.attributes["session.id"] == f"{name}-session"
            assert span.attributes["user.id"] == f"{name}-user"
    finally:
        provider.shutdown()


def test_concurrent_mobile_agents_keep_their_tracing_metadata(monkeypatch):
    from llama_index.core.workflow import StartEvent, step

    from mobilerun.agent.droid.droid_agent import MobileAgent
    from mobilerun.agent.droid.events import ResultEvent
    from mobilerun.config_manager.config_manager import (
        AgentConfig,
        MobileConfig,
        TelemetryConfig,
    )

    monkeypatch.setattr(tracing_setup, "_tracing_initialized", True)
    monkeypatch.setattr(tracing_setup, "_tracing_provider", "langfuse")
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(LangfuseSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    async def exercise():
        barrier = asyncio.Barrier(2)
        observations = {}

        class MetadataAgent(MobileAgent):
            @step
            async def start_handler(self, ev: StartEvent) -> ResultEvent:
                await barrier.wait()
                with tracer.start_as_current_span("MobileAgent.run") as span:
                    await asyncio.sleep(0)
                    current = langfuse_processor._current_agent.get()
                    assert current is self
                    assert (
                        await asyncio.to_thread(langfuse_processor._current_agent.get)
                        is self
                    )
                    name = self.shared_state.instruction
                    observations[name] = current.shared_state.instruction
                    span.set_attribute("test.agent", name)
                return ResultEvent(success=True, reason=name, steps=1)

        # Construct both agents in this task before scheduling either workflow.
        agents = [
            MetadataAgent(
                name,
                config=MobileConfig(
                    agent=AgentConfig(name="metadata-test-agent"),
                    telemetry=TelemetryConfig(enabled=False),
                    tracing=TracingConfig(enabled=True, provider="langfuse"),
                ),
                disable_validation=True,
            )
            for name in ("first", "second")
        ]
        caller_agent = langfuse_processor._current_agent.get()
        results = await asyncio.gather(agents[0].run(), agents[1].run())
        assert {result.reason for result in results} == {"first", "second"}
        assert observations == {"first": "first", "second": "second"}
        assert langfuse_processor._current_agent.get() is caller_agent

    try:
        asyncio.run(exercise())
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        for span in spans:
            metadata = json.loads(span.attributes["langfuse.observation.input"])
            assert metadata["goal"] == span.attributes["test.agent"]
    finally:
        provider.shutdown()


def test_mobile_agent_restores_context_if_workflow_start_fails(monkeypatch):
    from llama_index.core.workflow import Workflow
    from opentelemetry.context import get_current

    from mobilerun.agent.droid.droid_agent import MobileAgent
    from mobilerun.config_manager.config_manager import AgentConfig, MobileConfig

    monkeypatch.setattr(tracing_setup, "_tracing_initialized", True)
    monkeypatch.setattr(tracing_setup, "_tracing_provider", "langfuse")
    agent = MobileAgent(
        "Test context cleanup",
        config=MobileConfig(agent=AgentConfig(name="context-test-agent")),
    )
    monkeypatch.setattr(Workflow, "run", Mock(side_effect=RuntimeError("cannot start")))
    before = get_current()
    caller_agent = SimpleNamespace(name="caller")
    langfuse_processor._current_agent.set(caller_agent)
    with pytest.raises(RuntimeError, match="cannot start"):
        agent.run()
    assert get_current() is before
    assert langfuse_processor._current_agent.get() is caller_agent


@pytest.mark.parametrize("method", ["get_state", "get_final_state"])
def test_android_state_processing_runs_in_thread_with_unchanged_output(
    monkeypatch, method
):
    from mobilerun.tools.filters import ConciseFilter
    from mobilerun.tools.formatters import IndexedFormatter
    from mobilerun.tools.ui.provider import AndroidStateProvider

    data = {
        "a11y_tree": {
            "text": "Settings",
            "boundsInScreen": {"left": 0, "top": 0, "right": 1080, "bottom": 2400},
            "children": [
                {
                    "text": "Wi-Fi",
                    "boundsInScreen": {
                        "left": 10,
                        "top": 20,
                        "right": 200,
                        "bottom": 100,
                    },
                    "children": None,
                }
            ],
        },
        "device_context": {"screen_bounds": {"width": 1080, "height": 2400}},
        "phone_state": {
            "packageName": "com.android.settings",
            "currentApp": "Settings",
        },
    }
    driver = SimpleNamespace(get_ui_tree=AsyncMock(side_effect=lambda: deepcopy(data)))
    provider = AndroidStateProvider(
        driver, ConciseFilter(), IndexedFormatter(), vision_enabled=True
    )
    expected = provider._state_from_data(deepcopy(data))
    loop_thread = threading.get_ident()
    worker_threads = []
    original = AndroidStateProvider._state_from_data

    def process(self, raw):
        worker_threads.append(threading.get_ident())
        return original(self, raw)

    monkeypatch.setattr(AndroidStateProvider, "_state_from_data", process)
    actual = asyncio.run(getattr(provider, method)())
    assert vars(actual) == vars(expected)
    assert worker_threads and all(thread != loop_thread for thread in worker_threads)


@pytest.mark.parametrize("method", ["get_state", "get_final_state"])
@pytest.mark.parametrize("cancelled", [False, True], ids=["overlapping", "cancelled"])
def test_android_state_workers_cannot_corrupt_other_observations(
    monkeypatch, method, cancelled
):
    from mobilerun.tools.filters import ConciseFilter
    from mobilerun.tools.formatters import IndexedFormatter
    from mobilerun.tools.ui.provider import AndroidStateProvider

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    loop_thread = threading.get_ident()
    off_loop_writes = []

    def data(width, height):
        return {
            "a11y_tree": {
                "text": "Settings",
                "boundsInScreen": {
                    "left": 0,
                    "top": 0,
                    "right": width,
                    "bottom": height,
                },
                "children": [
                    {
                        "text": "Wi-Fi",
                        "boundsInScreen": {
                            "left": 0,
                            "top": 0,
                            "right": 1000,
                            "bottom": 1000,
                        },
                    }
                ],
            },
            "device_context": {"screen_bounds": {"width": width, "height": height}},
            "phone_state": {"packageName": "com.android.settings"},
        }

    large, small = data(4000, 8000), data(1000, 2000)

    class PausingFilter(ConciseFilter):
        def __init__(self):
            self.observed_sizes = []

        def filter(self, tree, context):
            self.observed_sizes.append(context["screen_bounds"]["width"])
            if cancelled and context["screen_bounds"]["width"] == 4000:
                entered.set()
                assert release.wait(5), "first worker was not released"
            return super().filter(tree, context)

    class PausingFormatter(IndexedFormatter):
        def __setattr__(self, name, value):
            if self is formatter and threading.get_ident() != loop_thread:
                off_loop_writes.append(name)
            super().__setattr__(name, value)

        def format(self, tree, phone_state):
            if not cancelled and self.screen_width == 4000:
                entered.set()
                assert release.wait(5), "first worker was not released"
            return super().format(tree, phone_state)

    # Keep synchronization events in method closures so per-call copies can
    # isolate configuration without copying threading locks.
    formatter = None
    formatter = PausingFormatter()
    tree_filter = PausingFilter()
    provider = AndroidStateProvider(
        SimpleNamespace(get_ui_tree=AsyncMock(side_effect=[large, small])),
        tree_filter,
        formatter,
        vision_enabled=True,
    )
    expected = [
        AndroidStateProvider(
            None, ConciseFilter(), IndexedFormatter(), vision_enabled=True
        )._state_from_data(deepcopy(raw))
        for raw in (large, small)
    ]
    original = AndroidStateProvider._state_from_data

    def process(self, raw):
        try:
            return original(self, raw)
        finally:
            if raw["device_context"]["screen_bounds"]["width"] == 4000:
                finished.set()

    monkeypatch.setattr(AndroidStateProvider, "_state_from_data", process)

    def assert_published(state):
        assert provider.model_screenshot_width == state.model_screenshot_width
        assert provider.model_screenshot_height == state.model_screenshot_height
        assert provider.resize_model_screenshot is state.coordinate_contract_active
        assert formatter.screen_width == state.screen_width
        assert formatter.screen_height == state.screen_height
        assert formatter.display_scale_x == state.coordinate_scale_x
        assert formatter.display_scale_y == state.coordinate_scale_y
        assert formatter.use_normalized == state.use_normalized

    async def exercise():
        first = asyncio.create_task(getattr(provider, method)())
        try:
            assert await asyncio.to_thread(
                entered.wait, 5
            ), "first worker did not pause"
            if cancelled:
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
            other_method = "get_final_state" if method == "get_state" else "get_state"
            fresh = await asyncio.wait_for(getattr(provider, other_method)(), 5)
            assert vars(fresh) == vars(expected[1])
            assert_published(fresh)
            release.set()
            if cancelled:
                assert await asyncio.to_thread(finished.wait, 5)
                await asyncio.sleep(0)
                assert_published(fresh)
            else:
                earlier = await asyncio.wait_for(first, 5)
                assert vars(earlier) == vars(expected[0])
                assert earlier.elements[1]["displayBounds"] == "0,0,256,256"
                assert_published(earlier)
            assert tree_filter.observed_sizes == []
            assert not off_loop_writes
        finally:
            release.set()
            await asyncio.gather(first, return_exceptions=True)

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "attributes",
    [
        {
            "resize_model_screenshot": True,
            "model_screenshot_width": 160,
            "model_screenshot_height": 320,
        },
        {
            "resize_model_screenshot": True,
            "model_screenshot_width": 160,
            "model_screenshot_height": 320,
            "model_screenshot_grid": False,
        },
        {"requires_coordinate_tools": True},
        {},
    ],
    ids=["pixel-grid", "normalized", "legacy", "unchanged"],
)
def test_async_screenshot_resize_preserves_output_and_offloads_work(
    monkeypatch, attributes
):
    from PIL import Image

    from mobilerun.tools.ui import provider as module

    loop_thread = threading.get_ident()

    class LoopStateProvider(SimpleNamespace):
        def __getattribute__(self, name):
            if name in {
                "model_screenshot_width",
                "model_screenshot_height",
                "model_screenshot_grid",
            }:
                assert threading.get_ident() == loop_thread
            return super().__getattribute__(name)

    provider = LoopStateProvider(**attributes)
    buffer = BytesIO()
    Image.new("RGB", (320, 640), "navy").save(buffer, format="PNG")
    screenshot = buffer.getvalue()
    should_resize = module.should_resize_model_screenshot(provider)
    original = module.resize_model_screenshot_with_grid
    expected = original(provider, screenshot) if should_resize else screenshot
    worker_threads = []

    def resize(state_provider, image):
        worker_threads.append(threading.get_ident())
        return original(state_provider, image)

    monkeypatch.setattr(module, "resize_model_screenshot_with_grid", resize)
    actual = asyncio.run(module.resize_model_screenshot_async(provider, screenshot))
    assert actual == expected
    if should_resize:
        assert worker_threads and all(
            thread != loop_thread for thread in worker_threads
        )
    else:
        assert actual is screenshot
        assert not worker_threads
