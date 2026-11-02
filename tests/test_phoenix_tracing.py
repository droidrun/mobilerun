import sys
from types import SimpleNamespace

import llama_index.core
from openinference.semconv.resource import ResourceAttributes

from mobilerun.agent.utils import tracing_setup
from mobilerun.telemetry import phoenix


def test_callback_factory_configures_otlp_export_and_instrumentor(
    monkeypatch,
):
    calls = SimpleNamespace()

    class FakeExporter:
        def __init__(self, endpoint):
            calls.exporter_endpoint = endpoint

    class FakeSpanProcessor:
        def __init__(self, exporter):
            calls.exporter = exporter

    class FakeTracerProvider:
        def __init__(self, resource):
            calls.resource = resource
            calls.created_provider = self

        def add_span_processor(self, processor):
            calls.processor = processor

    class FakeInstrumentor:
        def instrument(self, **kwargs):
            calls.instrument_kwargs = kwargs
            return "phoenix-handler"

    from openinference.instrumentation import llama_index as llama_index_instrumentation
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace import export as trace_export

    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", FakeExporter)
    monkeypatch.setattr(trace_export, "SimpleSpanProcessor", FakeSpanProcessor)
    monkeypatch.setattr(trace_sdk, "TracerProvider", FakeTracerProvider)
    monkeypatch.setattr(
        llama_index_instrumentation, "LlamaIndexInstrumentor", FakeInstrumentor
    )
    monkeypatch.setenv("phoenix_project_name", "compatibility-tests")

    supplied_provider = object()
    result = phoenix.arize_phoenix_callback_handler(
        endpoint="http://phoenix.example:6006",
        tracer_provider=supplied_provider,
        separate_trace_from_runtime_context=True,
    )

    assert result == "phoenix-handler"
    assert calls.exporter_endpoint == "http://phoenix.example:6006/v1/traces"
    assert calls.exporter is not None
    assert calls.processor is not None
    assert (
        calls.resource.attributes[ResourceAttributes.PROJECT_NAME]
        == "compatibility-tests"
    )
    assert calls.instrument_kwargs["tracer_provider"] is supplied_provider
    assert calls.instrument_kwargs["separate_trace_from_runtime_context"] is True
    assert calls.instrument_kwargs["config"].base64_image_max_length == 64_000_000


def test_phoenix_setup_checks_endpoint_and_installs_global_handler(monkeypatch):
    checked = []
    handler = object()

    monkeypatch.setenv("PHOENIX_URL", "http://phoenix.example:6006")
    monkeypatch.setattr(
        tracing_setup,
        "_check_phoenix_reachable",
        lambda endpoint: checked.append(endpoint) or True,
    )
    monkeypatch.setattr(phoenix, "arize_phoenix_callback_handler", lambda: handler)
    monkeypatch.setattr(llama_index.core, "global_handler", None)

    assert tracing_setup._setup_phoenix_tracing() is True
    assert checked == ["http://phoenix.example:6006"]
    assert llama_index.core.global_handler is handler


def test_phoenix_setup_stays_disabled_when_endpoint_is_unreachable(monkeypatch):
    monkeypatch.setenv("PHOENIX_URL", "http://phoenix.example:6006")
    monkeypatch.setattr(
        tracing_setup, "_check_phoenix_reachable", lambda _endpoint: False
    )
    monkeypatch.setattr(
        phoenix,
        "arize_phoenix_callback_handler",
        lambda: (_ for _ in ()).throw(AssertionError("handler must not be created")),
    )

    assert tracing_setup._setup_phoenix_tracing() is False


def test_phoenix_setup_reports_reinstall_when_dependencies_are_missing(monkeypatch):
    warnings = []
    monkeypatch.setitem(sys.modules, "mobilerun.telemetry.phoenix", None)
    monkeypatch.setattr(tracing_setup.logger, "warning", warnings.append)

    assert tracing_setup._setup_phoenix_tracing() is False

    assert len(warnings) == 1
    assert "[phoenix]" not in warnings[0]
    assert "uv tool install --reinstall mobilerun`\n" in warnings[0]
    assert "uv pip install --reinstall mobilerun`\n" in warnings[0]


def test_phoenix_setup_reports_reinstall_when_handler_imports_fail(monkeypatch):
    warnings = []
    global_handler = object()

    def missing_instrumentation():
        raise ModuleNotFoundError("No module named 'openinference'")

    monkeypatch.setattr(
        tracing_setup, "_check_phoenix_reachable", lambda _endpoint: True
    )
    monkeypatch.setattr(
        phoenix, "arize_phoenix_callback_handler", missing_instrumentation
    )
    monkeypatch.setattr(llama_index.core, "global_handler", global_handler)
    monkeypatch.setattr(tracing_setup.logger, "warning", warnings.append)

    assert tracing_setup._setup_phoenix_tracing() is False

    assert len(warnings) == 1
    assert "uv tool install --reinstall mobilerun`\n" in warnings[0]
    assert llama_index.core.global_handler is global_handler
