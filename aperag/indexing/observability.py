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

"""Observability primitives — celery T1.5.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §J.1, the
indexing redesign emits four service-level indicators (SLIs) that
operators alert on:

- ``index_lag_seconds`` (gauge) — age of the oldest ``RUNNING`` row in
  ``document_index`` (queue + work latency).
- ``index_failure_rate`` (counter pair) — successes vs failures of
  ``sync_<modality>`` calls.
- ``queue_depth`` (gauge) — count of ``PENDING`` rows in
  ``document_index``.
- ``worker_utilization`` (gauge in [0, 1]) — fraction of worker pool
  slots currently running a task.

This module defines a minimal :class:`MetricsEmitter` protocol plus a
process-local :class:`InMemoryMetricsEmitter` (test fixture), a no-op
:class:`NoopMetricsEmitter` (default for ``INDEXING_METRICS_EMITTER=noop``
single-machine deployments where OTLP is not wired), and the production
:class:`OTLPMetricsEmitter` (wires the four §J.1 SLIs onto the SDK
``MeterProvider`` configured by :mod:`aperag.observability.metrics`,
selected by ``INDEXING_METRICS_EMITTER=otlp``).

The emit helpers (``emit_index_lag`` / ``emit_index_failure`` /
``emit_queue_depth`` / ``emit_worker_utilization``) are the canonical
call shapes that downstream T2 worker / reconciler code uses.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from aperag.indexing.models import Modality

logger = logging.getLogger(__name__)


# Metric names — exposed as constants so tests + downstream code can
# reference them by symbol rather than re-typing the string literal.
INDEX_LAG_METRIC = "indexing.index_lag_seconds"
INDEX_FAILURE_METRIC = "indexing.index_failure_total"
INDEX_SUCCESS_METRIC = "indexing.index_success_total"
QUEUE_DEPTH_METRIC = "indexing.queue_depth"
WORKER_UTILIZATION_METRIC = "indexing.worker_utilization"


@runtime_checkable
class MetricsEmitter(Protocol):
    """Minimal sink surface for the four indexing SLIs.

    Production binding emits via OTLP; tests bind to
    :class:`InMemoryMetricsEmitter`. The two methods are everything
    the §J.1 SLIs need.
    """

    def gauge(self, *, name: str, value: float, attributes: dict[str, str] | None = None) -> None:
        """Record a gauge sample."""

    def counter(self, *, name: str, value: float = 1.0, attributes: dict[str, str] | None = None) -> None:
        """Increment a counter."""


class NoopMetricsEmitter:
    """No-op emitter — the default ``INDEXING_METRICS_EMITTER=noop`` binding.

    Suitable for single-machine deployments without an OTLP collector
    (Tier 1 §L) and for unit tests that don't care about metric output.
    Production multi-pod deployments **must** opt into
    :class:`OTLPMetricsEmitter` via ``INDEXING_METRICS_EMITTER=otlp`` so
    the §J.1 SLIs reach the collector — silently dropping metrics on
    Tier 2/3 means operators lose the queue-backlog / failure-rate
    signals they alert on.
    """

    def gauge(
        self,
        *,
        name: str,
        value: float,
        attributes: dict[str, str] | None = None,
    ) -> None:  # pragma: no cover - trivial
        return None

    def counter(
        self,
        *,
        name: str,
        value: float = 1.0,
        attributes: dict[str, str] | None = None,
    ) -> None:  # pragma: no cover - trivial
        return None


class OTLPMetricsEmitter:
    """Production emitter — materialises SDK instruments on the global
    OpenTelemetry ``MeterProvider`` and forwards every ``gauge`` /
    ``counter`` call to that provider's exporter.

    The constructor pulls ``opentelemetry.metrics.get_meter`` lazily so
    importing this module never forces an OTLP install. When the SDK
    packages or :func:`aperag.observability.metrics.init_metrics_provider`
    have not run, the global provider is the no-op
    ``ProxyMeterProvider`` and every call here is silently dropped —
    the operator-facing diagnosis is therefore "your OTLP endpoint is
    not configured" rather than a crash, which matches how the rest of
    :mod:`aperag.observability` degrades.

    Instruments are cached on the instance so repeated emits for the
    same metric name share one ``Counter`` / ``Gauge`` instrument
    handle (mirrors the pattern in
    :mod:`aperag.observability.metrics.record_counter`).
    """

    def __init__(
        self,
        meter_name: str = "aperag.indexing",
        *,
        meter: object | None = None,
    ) -> None:
        if meter is not None:
            # Tests pass a meter sourced from an isolated SDK
            # ``MeterProvider`` (with an :class:`InMemoryMetricReader`)
            # so they don't have to mutate the process-global provider
            # — ``opentelemetry.metrics.set_meter_provider`` only takes
            # effect once per process and cannot be reset between tests.
            self._meter = meter
        else:
            try:
                from opentelemetry import metrics as otel_metrics
            except ImportError as exc:  # pragma: no cover - depends on runtime install
                raise RuntimeError(
                    "OTLPMetricsEmitter requires the opentelemetry-api package; "
                    "fall back to NoopMetricsEmitter when OTLP is not installed."
                ) from exc
            self._meter = otel_metrics.get_meter(meter_name)
        self._gauges: dict[str, object] = {}
        self._counters: dict[str, object] = {}

    def gauge(
        self,
        *,
        name: str,
        value: float,
        attributes: dict[str, str] | None = None,
    ) -> None:
        instrument = self._gauges.get(name)
        if instrument is None:
            try:
                instrument = self._meter.create_gauge(name)
            except Exception:
                logger.debug("OTLP gauge create failed for %s", name, exc_info=True)
                return
            self._gauges[name] = instrument
        try:
            instrument.set(float(value), dict(attributes or {}))
        except Exception:
            logger.debug("OTLP gauge set failed for %s", name, exc_info=True)

    def counter(
        self,
        *,
        name: str,
        value: float = 1.0,
        attributes: dict[str, str] | None = None,
    ) -> None:
        instrument = self._counters.get(name)
        if instrument is None:
            try:
                instrument = self._meter.create_counter(name)
            except Exception:
                logger.debug("OTLP counter create failed for %s", name, exc_info=True)
                return
            self._counters[name] = instrument
        try:
            instrument.add(float(value), dict(attributes or {}))
        except Exception:
            logger.debug("OTLP counter add failed for %s", name, exc_info=True)


class InMemoryMetricsEmitter:
    """Process-local emitter for unit tests.

    Holds the most recent gauge value per ``(name, attributes)`` and
    a running total per counter. Tests assert against the recorded
    state directly.
    """

    def __init__(self) -> None:
        self.gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self.counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}

    @staticmethod
    def _key(name: str, attributes: dict[str, str] | None) -> tuple[str, tuple[tuple[str, str], ...]]:
        attr_tuple = tuple(sorted((attributes or {}).items()))
        return (name, attr_tuple)

    def gauge(
        self,
        *,
        name: str,
        value: float,
        attributes: dict[str, str] | None = None,
    ) -> None:
        self.gauges[self._key(name, attributes)] = float(value)

    def counter(
        self,
        *,
        name: str,
        value: float = 1.0,
        attributes: dict[str, str] | None = None,
    ) -> None:
        key = self._key(name, attributes)
        self.counters[key] = self.counters.get(key, 0.0) + float(value)


# ---------------------------------------------------------------------
# Emit helpers — the canonical call shapes for §J.1 SLIs
# ---------------------------------------------------------------------


def emit_index_lag(
    emitter: MetricsEmitter,
    *,
    seconds: float,
    modality: Modality | str | None = None,
) -> None:
    """Emit the ``index_lag_seconds`` gauge.

    The reconciler computes ``seconds`` as ``now - oldest RUNNING
    document_index.last_heartbeat`` (or ``created_at`` when no
    heartbeat yet). When the modality is known, attach it as an
    attribute so dashboards can break down per-modality lag.
    """
    attributes = None
    if modality is not None:
        modality_value = modality.value if isinstance(modality, Modality) else str(modality)
        attributes = {"modality": modality_value}
    emitter.gauge(name=INDEX_LAG_METRIC, value=float(seconds), attributes=attributes)


def emit_index_failure(
    emitter: MetricsEmitter,
    *,
    modality: Modality | str,
    error_kind: str = "unknown",
) -> None:
    """Emit a single increment of the ``index_failure_total`` counter."""
    modality_value = modality.value if isinstance(modality, Modality) else str(modality)
    emitter.counter(
        name=INDEX_FAILURE_METRIC,
        value=1.0,
        attributes={"modality": modality_value, "error_kind": error_kind},
    )


def emit_index_success(
    emitter: MetricsEmitter,
    *,
    modality: Modality | str,
) -> None:
    """Emit a single increment of the ``index_success_total`` counter.

    Paired with :func:`emit_index_failure` so dashboards can compute
    the failure rate as ``failures / (successes + failures)`` per
    modality.
    """
    modality_value = modality.value if isinstance(modality, Modality) else str(modality)
    emitter.counter(
        name=INDEX_SUCCESS_METRIC,
        value=1.0,
        attributes={"modality": modality_value},
    )


def emit_queue_depth(
    emitter: MetricsEmitter,
    *,
    depth: int,
    modality: Modality | str | None = None,
) -> None:
    """Emit the ``queue_depth`` gauge.

    Reconciler counts ``PENDING`` rows in ``document_index`` per
    modality (or unscoped) and emits this gauge once per cycle.
    """
    attributes = None
    if modality is not None:
        modality_value = modality.value if isinstance(modality, Modality) else str(modality)
        attributes = {"modality": modality_value}
    emitter.gauge(name=QUEUE_DEPTH_METRIC, value=float(depth), attributes=attributes)


def emit_worker_utilization(
    emitter: MetricsEmitter,
    *,
    busy: int,
    capacity: int,
    modality: Modality | str | None = None,
) -> None:
    """Emit the ``worker_utilization`` gauge as ``busy / capacity``.

    Capacity is the worker-pool size for that modality (or unscoped).
    Capacity 0 → emit 0 (nothing to do).
    """
    if capacity <= 0:
        utilization = 0.0
    else:
        utilization = max(0.0, min(1.0, float(busy) / float(capacity)))
    attributes = None
    if modality is not None:
        modality_value = modality.value if isinstance(modality, Modality) else str(modality)
        attributes = {"modality": modality_value}
    emitter.gauge(
        name=WORKER_UTILIZATION_METRIC,
        value=utilization,
        attributes=attributes,
    )


__all__ = [
    "MetricsEmitter",
    "NoopMetricsEmitter",
    "OTLPMetricsEmitter",
    "InMemoryMetricsEmitter",
    "INDEX_LAG_METRIC",
    "INDEX_FAILURE_METRIC",
    "INDEX_SUCCESS_METRIC",
    "QUEUE_DEPTH_METRIC",
    "WORKER_UTILIZATION_METRIC",
    "emit_index_lag",
    "emit_index_failure",
    "emit_index_success",
    "emit_queue_depth",
    "emit_worker_utilization",
]
