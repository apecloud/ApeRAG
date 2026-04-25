# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenTelemetry tracing initialization and helpers."""

from __future__ import annotations

import functools
import logging
import os
from contextlib import nullcontext
from typing import Any, Optional

from aperag.observability.config import ObservabilityConfig
from aperag.observability.metrics import configure_metrics

logger = logging.getLogger(__name__)

try:
    from opentelemetry import context as otel_context
    from opentelemetry import propagate, trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ParentBased, TraceIdRatioBased
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState, set_span_in_context

    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on runtime package availability
    OTEL_AVAILABLE = False


_initialized = False
_sqlalchemy_instrumented = False
_fastapi_apps_instrumented: set[int] = set()


def is_available() -> bool:
    return OTEL_AVAILABLE


def is_initialized() -> bool:
    return _initialized


def init_tracing(config: ObservabilityConfig) -> bool:
    """Initialize tracing once for the current process."""
    global _initialized

    if _initialized:
        return True

    if not OTEL_AVAILABLE:
        if config.enabled:
            logger.warning("OpenTelemetry packages are not available; tracing is disabled")
        return False

    if config.mode == "off":
        _initialized = True
        return True

    try:
        sampler = ALWAYS_OFF if config.mode == "local" and not config.console_enabled else _build_sampler(config)
        provider = TracerProvider(resource=_build_resource(config), sampler=sampler)

        exporters_added = 0
        if config.console_enabled:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            exporters_added += 1

        if config.uses_otlp:
            exporter = _build_otlp_exporter(config)
            if exporter is not None:
                provider.add_span_processor(BatchSpanProcessor(exporter))
                exporters_added += 1

        trace.set_tracer_provider(provider)
        _initialized = True
        logger.info(
            "observability tracing initialized",
            extra={
                "operation": "observability.tracing.init",
                "outcome": "success",
                "observability_mode": config.mode,
                "exporters": exporters_added,
            },
        )
        return True
    except Exception:
        logger.exception(
            "failed to initialize observability tracing",
            extra={"operation": "observability.tracing.init", "outcome": "failure"},
        )
        return False


def configure_process_observability(config: ObservabilityConfig) -> bool:
    """Configure process-wide tracing and metrics."""
    global _sqlalchemy_instrumented

    configure_metrics(config)
    initialized = init_tracing(config)
    if (
        initialized
        and OTEL_AVAILABLE
        and config.enable_sqlalchemy
        and config.mode != "off"
        and not _sqlalchemy_instrumented
    ):
        try:
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

            SQLAlchemyInstrumentor().instrument()
            _sqlalchemy_instrumented = True
        except Exception:
            logger.exception(
                "failed to instrument SQLAlchemy",
                extra={"operation": "observability.sqlalchemy.instrument", "outcome": "failure"},
            )
    return initialized


def configure_fastapi(app: Any, config: ObservabilityConfig) -> bool:
    """Instrument one FastAPI app instance explicitly."""
    if not OTEL_AVAILABLE or config.mode == "off" or not config.enable_fastapi:
        return False
    app_id = id(app)
    if app_id in _fastapi_apps_instrumented:
        return True
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        _fastapi_apps_instrumented.add(app_id)
        return True
    except Exception:
        logger.exception(
            "failed to instrument FastAPI app",
            extra={"operation": "observability.fastapi.instrument", "outcome": "failure"},
        )
        return False


def shutdown_tracing() -> None:
    """Flush and shut down the configured tracer provider when supported."""
    if not OTEL_AVAILABLE:
        return
    try:
        provider = trace.get_tracer_provider()
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        logger.debug("failed to shut down tracer provider", exc_info=True)


def _build_resource(config: ObservabilityConfig):
    attributes: dict[str, Any] = {
        "service.name": config.service_name,
        "service.version": config.service_version,
        "service.instance.id": service_instance_id(),
    }
    if config.environment:
        attributes["deployment.environment"] = config.environment
    attributes.update(config.resource_attributes)
    return Resource.create(attributes)


def _build_sampler(config: ObservabilityConfig):
    ratio = min(max(config.sample_ratio, 0.0), 1.0)
    return ParentBased(TraceIdRatioBased(ratio))


def _build_otlp_exporter(config: ObservabilityConfig):
    if not config.otlp_endpoint:
        logger.warning(
            "OTLP observability mode selected without OTEL_EXPORTER_OTLP_ENDPOINT; remote tracing disabled",
            extra={"operation": "observability.tracing.init", "outcome": "skipped"},
        )
        return None

    protocol = config.otlp_protocol.lower()
    if protocol in {"http/protobuf", "http"}:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter(endpoint=_trace_http_endpoint(config.otlp_endpoint), headers=config.otlp_headers)

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=config.otlp_endpoint, headers=config.otlp_headers or None)


def _trace_http_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1/traces"):
        return endpoint
    return f"{endpoint}/v1/traces"


def get_tracer(name: str):
    if not OTEL_AVAILABLE:
        return None
    return trace.get_tracer(name)


def start_span(name: str, tracer_name: str = "aperag", **attributes):
    tracer = get_tracer(tracer_name)
    if tracer is None:
        return nullcontext()

    span_cm = tracer.start_as_current_span(name)

    class SpanWrapper:
        def __enter__(self):
            span = span_cm.__enter__()
            add_attributes(**attributes)
            return span

        def __exit__(self, exc_type, exc_val, exc_tb):
            return span_cm.__exit__(exc_type, exc_val, exc_tb)

    return SpanWrapper()


def trace_function(name: Optional[str] = None):
    """Trace a synchronous function with a span."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or func.__qualname__
            with start_span(span_name, tracer_name=func.__module__, function=func.__name__, module=func.__module__):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_async_function(name: Optional[str] = None):
    """Trace an async function with a span."""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            span_name = name or func.__qualname__
            with start_span(span_name, tracer_name=func.__module__, function=func.__name__, module=func.__module__):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def add_attributes(**attributes: Any) -> None:
    if not OTEL_AVAILABLE:
        return
    try:
        span = trace.get_current_span()
        if span and span.is_recording():
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
    except Exception:
        logger.debug("failed to add span attributes", exc_info=True)


add_span_attributes = add_attributes


def get_current_trace_info() -> tuple[Optional[str], Optional[str]]:
    if not OTEL_AVAILABLE:
        return None, None
    try:
        span = trace.get_current_span()
        context = span.get_span_context()
        if context and context.is_valid:
            return format(context.trace_id, "032x"), format(context.span_id, "016x")
    except Exception:
        return None, None
    return None, None


def inject_carrier(carrier: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    carrier = carrier or {}
    if OTEL_AVAILABLE:
        propagate.inject(carrier)
    return carrier


def extract_context(carrier: Optional[dict[str, Any]] = None):
    if not OTEL_AVAILABLE:
        return None
    return propagate.extract(carrier or {})


def attach_context_from_carrier(carrier: Optional[dict[str, Any]] = None):
    if not OTEL_AVAILABLE:
        return None
    return otel_context.attach(extract_context(carrier))


def detach_context(token) -> None:
    if OTEL_AVAILABLE and token is not None:
        otel_context.detach(token)


def make_span_context(trace_id: str, span_id: Optional[str] = None):
    """Build a non-recording span context for log-only correlation."""
    if not OTEL_AVAILABLE:
        return None
    try:
        span_context = SpanContext(
            trace_id=int(trace_id, 16),
            span_id=int(span_id or "0" * 15 + "1", 16),
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
            trace_state=TraceState(),
        )
        return set_span_in_context(NonRecordingSpan(span_context))
    except Exception:
        logger.debug("invalid trace/span context", exc_info=True)
        return None


def service_instance_id() -> str:
    return os.getenv("HOSTNAME") or os.getenv("NODE_IP") or ""
