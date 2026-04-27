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

"""Integration tests for ``OTLPMetricsEmitter`` — Wave 4 T6.

Pin the Wave 4 #8 production-readiness invariant: when
``INDEXING_METRICS_EMITTER=otlp`` (production multi-pod per design pack
§J.1), :class:`OTLPMetricsEmitter` materialises real SDK instruments on
the configured ``MeterProvider`` and forwards every ``gauge`` /
``counter`` call to the OTLP exporter.

The exporter side is verified with :class:`InMemoryMetricReader` so the
test does not need a live OTLP collector — it is exercising the same
SDK path the production exporter uses (``MeterProvider`` →
``MetricReader`` → ``Metric`` data) just with the in-memory reader
substituted for the network exporter.

The emitter accepts an injectable ``meter`` for tests because
``opentelemetry.metrics.set_meter_provider`` is one-shot per process —
once any real provider is installed, it cannot be replaced, so tests
that need isolated readers must source meters from their own
``MeterProvider`` instances and pass them in directly.
"""

from __future__ import annotations

import importlib

import pytest

from aperag.indexing.models import Modality
from aperag.indexing.observability import (
    INDEX_FAILURE_METRIC,
    INDEX_SUCCESS_METRIC,
    QUEUE_DEPTH_METRIC,
    WORKER_UTILIZATION_METRIC,
    OTLPMetricsEmitter,
    emit_index_failure,
    emit_index_success,
    emit_queue_depth,
    emit_worker_utilization,
)


@pytest.fixture
def isolated_provider():
    """Per-test SDK ``MeterProvider`` with an
    :class:`InMemoryMetricReader`. Returned tuple is
    ``(provider, reader)`` — tests source a meter from the provider
    via ``provider.get_meter(...)`` and read back the recorded
    samples from the reader.
    """

    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.resources import Resource

    reader = InMemoryMetricReader()
    provider = MeterProvider(
        resource=Resource.create({"service.name": "aperag-test"}),
        metric_readers=[reader],
    )
    try:
        yield provider, reader
    finally:
        provider.shutdown()


def _collect_samples(reader) -> dict[str, list[tuple[dict, float]]]:
    """Drain the reader once and bucket all data points by metric name.

    ``InMemoryMetricReader.get_metrics_data()`` collects + resets — a
    second call returns ``None`` because the first one consumed every
    point. Tests therefore drain once and inspect the snapshot, rather
    than calling ``_samples_for(reader, name)`` multiple times.
    """

    data = reader.get_metrics_data()
    samples: dict[str, list[tuple[dict, float]]] = {}
    if data is None:
        return samples
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                bucket = samples.setdefault(metric.name, [])
                for point in metric.data.data_points:
                    attrs = dict(point.attributes or {})
                    value = getattr(point, "value", None)
                    if value is None:
                        continue
                    bucket.append((attrs, float(value)))
    return samples


def test_counter_round_trip(isolated_provider):
    """``counter`` calls land on the SDK ``MeterProvider`` and the
    in-memory reader sees the running total, attribute-keyed.
    """
    provider, reader = isolated_provider
    emitter = OTLPMetricsEmitter(meter=provider.get_meter("aperag.indexing.test"))

    emit_index_success(emitter, modality=Modality.VECTOR)
    emit_index_success(emitter, modality=Modality.VECTOR)
    emit_index_failure(emitter, modality=Modality.VECTOR, error_kind="parse")

    samples = _collect_samples(reader)

    assert samples[INDEX_SUCCESS_METRIC] == [({"modality": "vector"}, 2.0)]
    assert samples[INDEX_FAILURE_METRIC] == [({"modality": "vector", "error_kind": "parse"}, 1.0)]


def test_gauge_round_trip(isolated_provider):
    """``gauge`` calls update an SDK ``Gauge`` instrument — the latest
    value per attribute set wins, mirroring the
    :class:`InMemoryMetricsEmitter` semantics.
    """
    provider, reader = isolated_provider
    emitter = OTLPMetricsEmitter(meter=provider.get_meter("aperag.indexing.test"))

    emit_queue_depth(emitter, depth=3, modality=Modality.FULLTEXT)
    emit_queue_depth(emitter, depth=11, modality=Modality.FULLTEXT)
    emit_worker_utilization(emitter, busy=2, capacity=8, modality=Modality.GRAPH)

    samples = _collect_samples(reader)

    assert samples[QUEUE_DEPTH_METRIC] == [({"modality": "fulltext"}, 11.0)]
    # Utilization = 2 / 8 = 0.25
    assert samples[WORKER_UTILIZATION_METRIC] == [({"modality": "graph"}, 0.25)]


def test_per_modality_isolation(isolated_provider):
    """Two modality attributes on the same metric name fan out to two
    distinct data points so operators get per-modality dashboards.
    """
    provider, reader = isolated_provider
    emitter = OTLPMetricsEmitter(meter=provider.get_meter("aperag.indexing.test"))

    emit_queue_depth(emitter, depth=4, modality=Modality.VECTOR)
    emit_queue_depth(emitter, depth=7, modality=Modality.SUMMARY)

    samples = sorted(
        _collect_samples(reader)[QUEUE_DEPTH_METRIC],
        key=lambda pair: pair[0]["modality"],
    )
    assert samples == [
        ({"modality": "summary"}, 7.0),
        ({"modality": "vector"}, 4.0),
    ]


def test_instruments_are_reused_per_metric_name(isolated_provider):
    """Repeated ``counter`` / ``gauge`` calls for the same name reuse
    one SDK instrument handle — instrument creation is idempotent and
    not per-call (mirrors the pattern in
    ``aperag.observability.metrics.record_counter``).
    """
    provider, _ = isolated_provider
    emitter = OTLPMetricsEmitter(meter=provider.get_meter("aperag.indexing.test"))

    emitter.counter(name=INDEX_SUCCESS_METRIC, attributes={"modality": "vector"})
    emitter.counter(name=INDEX_SUCCESS_METRIC, attributes={"modality": "fulltext"})
    emitter.gauge(name=QUEUE_DEPTH_METRIC, value=2.0)
    emitter.gauge(name=QUEUE_DEPTH_METRIC, value=4.0)

    assert list(emitter._counters.keys()) == [INDEX_SUCCESS_METRIC]
    assert list(emitter._gauges.keys()) == [QUEUE_DEPTH_METRIC]


def test_init_metrics_provider_endpoint_required():
    """``init_metrics_provider`` short-circuits when the OTLP endpoint
    is not configured. Operators in ``mode=otlp`` without an endpoint
    get a warning + the app keeps booting against the no-op proxy
    provider — silent enough that the indexing emitter degrades to
    no-ops, loud enough that the warning is in the log.
    """
    # Reload so ``_meter_provider_initialized`` resets to ``False``.
    from aperag.observability import metrics as obs_metrics

    importlib.reload(obs_metrics)

    from aperag.observability.config import ObservabilityConfig

    no_endpoint = ObservabilityConfig(mode="otlp", otlp_endpoint=None)
    assert obs_metrics.init_metrics_provider(no_endpoint) is False
    assert obs_metrics._meter_provider_initialized is False


def test_init_metrics_provider_idempotent():
    """``init_metrics_provider`` is safe to call multiple times. Once
    the SDK ``MeterProvider`` is installed at app start, subsequent
    calls (e.g. from a hot-reload code path) return ``True`` without
    re-initialising.

    NOTE: ``set_meter_provider`` is one-shot at the global level, so
    we exercise the idempotence guard explicitly rather than swapping
    the global provider mid-test.
    """
    from aperag.observability import metrics as obs_metrics
    from aperag.observability.config import ObservabilityConfig

    importlib.reload(obs_metrics)

    config = ObservabilityConfig(
        mode="otlp",
        otlp_endpoint="http://127.0.0.1:4318",
        otlp_protocol="http/protobuf",
    )
    first = obs_metrics.init_metrics_provider(config)
    second = obs_metrics.init_metrics_provider(config)
    assert first is True
    assert second is True
    assert obs_metrics._meter_provider_initialized is True

    obs_metrics.shutdown_metrics_provider()


def test_otlp_emitter_default_meter_resolves_global():
    """Without an injected meter, :class:`OTLPMetricsEmitter` resolves
    the global ``opentelemetry.metrics.get_meter("aperag.indexing")``
    so production lifespan dispatch (which does not pass ``meter=``)
    binds to whichever ``MeterProvider`` is installed at app start.
    """
    emitter = OTLPMetricsEmitter()
    # The default proxy provider returns a non-None meter; we don't
    # assert on its concrete type because it varies by SDK init order.
    # The ``gauge`` / ``counter`` calls below must not crash even
    # when the proxy is in place.
    emitter.gauge(name=QUEUE_DEPTH_METRIC, value=1.0)
    emitter.counter(name=INDEX_SUCCESS_METRIC, value=1.0, attributes={"modality": "vector"})
    assert emitter._meter is not None
