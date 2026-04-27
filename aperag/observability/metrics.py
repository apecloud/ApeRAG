"""Small metrics facade.

Business code can depend on this module without knowing whether the current
process exports metrics. In ``local`` and ``off`` modes these functions are
cheap no-ops.

When ``mode in {otlp, collector}`` and an OTLP endpoint is set,
:func:`configure_metrics` also initialises an SDK ``MeterProvider`` with a
``PeriodicExportingMetricReader`` wrapping ``OTLPMetricExporter`` so the
``aperag.indexing.OTLPMetricsEmitter`` (and ``record_counter`` /
``record_histogram`` here) actually ship samples to the collector. Without
this initialisation OpenTelemetry's global ``MeterProvider`` is the no-op
``ProxyMeterProvider`` and every metric call silently drops.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Mapping, Optional

from .config import ObservabilityConfig

logger = logging.getLogger(__name__)

_config: Optional[ObservabilityConfig] = None
_counters: dict[str, object] = {}
_histograms: dict[str, object] = {}
_meter_provider_initialized = False


def configure_metrics(config: ObservabilityConfig) -> None:
    global _config
    _config = config
    if config.metrics_enabled:
        init_metrics_provider(config)


def init_metrics_provider(config: ObservabilityConfig) -> bool:
    """Install an SDK ``MeterProvider`` with an OTLP exporter.

    Idempotent — returns ``True`` after first successful init and on
    every subsequent call. Returns ``False`` when the OpenTelemetry
    SDK packages are not importable, or when the exporter could not
    be constructed (logged + swallowed so the indexing emitter falls
    back to a no-op rather than crashing the app).
    """

    global _meter_provider_initialized

    if _meter_provider_initialized:
        return True

    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
    except ImportError:  # pragma: no cover - depends on runtime package availability
        logger.warning("OpenTelemetry SDK metrics packages are not available; OTLP metrics disabled")
        return False

    if not config.metrics_enabled:
        return False

    exporter = _build_otlp_metric_exporter(config)
    if exporter is None:
        return False

    resource_attributes: dict[str, object] = {
        "service.name": config.service_name,
        "service.version": config.service_version,
    }
    if config.environment:
        resource_attributes["deployment.environment"] = config.environment
    resource_attributes.update(config.resource_attributes)

    try:
        reader = PeriodicExportingMetricReader(exporter)
        provider = MeterProvider(
            resource=Resource.create(resource_attributes),
            metric_readers=[reader],
        )
        metrics.set_meter_provider(provider)
        # Existing module-level lru_cache for ``_meter()`` was sized
        # to one entry against the global proxy MeterProvider; clear it
        # so the next call binds to the SDK provider we just installed.
        _meter.cache_clear()
        _meter_provider_initialized = True
        logger.info(
            "observability metrics initialized",
            extra={
                "operation": "observability.metrics.init",
                "outcome": "success",
                "observability_mode": config.mode,
            },
        )
        return True
    except Exception:
        logger.exception(
            "failed to initialize observability metrics",
            extra={"operation": "observability.metrics.init", "outcome": "failure"},
        )
        return False


def shutdown_metrics_provider() -> None:
    """Flush + shut down the configured SDK MeterProvider on exit."""
    if not _meter_provider_initialized:
        return
    try:
        from opentelemetry import metrics
    except ImportError:  # pragma: no cover
        return
    try:
        provider = metrics.get_meter_provider()
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        logger.debug("failed to shut down meter provider", exc_info=True)


def _build_otlp_metric_exporter(config: ObservabilityConfig):
    if not config.otlp_endpoint:
        logger.warning(
            "OTLP observability mode selected without OTEL_EXPORTER_OTLP_ENDPOINT; remote metrics disabled",
            extra={"operation": "observability.metrics.init", "outcome": "skipped"},
        )
        return None
    protocol = (config.otlp_protocol or "http/protobuf").lower()
    headers = config.otlp_headers or None
    if protocol in {"http/protobuf", "http"}:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        endpoint = _metric_http_endpoint(config.otlp_endpoint)
        return OTLPMetricExporter(endpoint=endpoint, headers=headers)
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

    return OTLPMetricExporter(endpoint=config.otlp_endpoint, headers=headers)


def _metric_http_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1/metrics"):
        return endpoint
    return f"{endpoint}/v1/metrics"


@lru_cache(maxsize=1)
def _meter():
    try:
        from opentelemetry import metrics
    except ImportError:
        return None
    return metrics.get_meter("aperag")


def get_meter():
    return _meter()


def record_counter(name: str, value: int = 1, attributes: Mapping[str, object] | None = None) -> None:
    if _config is None or not _config.metrics_enabled:
        return
    meter = _meter()
    if meter is None:
        return
    try:
        counter = _counters.get(name)
        if counter is None:
            counter = meter.create_counter(name)
            _counters[name] = counter
        counter.add(value, dict(attributes or {}))
    except Exception as exc:
        logger.debug("Failed to record counter %s: %s", name, exc)


def record_histogram(name: str, value: float, attributes: Mapping[str, object] | None = None) -> None:
    if _config is None or not _config.metrics_enabled:
        return
    meter = _meter()
    if meter is None:
        return
    try:
        histogram = _histograms.get(name)
        if histogram is None:
            histogram = meter.create_histogram(name)
            _histograms[name] = histogram
        histogram.record(value, dict(attributes or {}))
    except Exception as exc:
        logger.debug("Failed to record histogram %s: %s", name, exc)
