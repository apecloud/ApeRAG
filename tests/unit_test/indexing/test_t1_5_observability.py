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

"""T1.5 Observability primitives contract tests.

Locks the §J.1 SLI emission shapes and the four ``emit_*`` helper
contracts. T1.5 ships only the in-memory + no-op emitters; production
OTLP wiring lands in T2.x.
"""

from __future__ import annotations

from aperag.indexing import (
    INDEX_FAILURE_METRIC,
    INDEX_LAG_METRIC,
    INDEX_SUCCESS_METRIC,
    QUEUE_DEPTH_METRIC,
    WORKER_UTILIZATION_METRIC,
    InMemoryMetricsEmitter,
    Modality,
    NoopMetricsEmitter,
    emit_index_failure,
    emit_index_lag,
    emit_index_success,
    emit_queue_depth,
    emit_worker_utilization,
)


def test_emit_index_lag_records_gauge_with_modality_attribute():
    emitter = InMemoryMetricsEmitter()
    emit_index_lag(emitter, seconds=12.5, modality=Modality.VECTOR)
    key = (INDEX_LAG_METRIC, (("modality", "vector"),))
    assert emitter.gauges[key] == 12.5


def test_emit_index_lag_emits_unscoped_when_modality_none():
    emitter = InMemoryMetricsEmitter()
    emit_index_lag(emitter, seconds=4.0)
    key = (INDEX_LAG_METRIC, ())
    assert emitter.gauges[key] == 4.0


def test_emit_index_failure_increments_counter_with_error_kind():
    emitter = InMemoryMetricsEmitter()
    emit_index_failure(emitter, modality=Modality.GRAPH, error_kind="nebula_timeout")
    emit_index_failure(emitter, modality=Modality.GRAPH, error_kind="nebula_timeout")
    key = (INDEX_FAILURE_METRIC, (("error_kind", "nebula_timeout"), ("modality", "graph")))
    assert emitter.counters[key] == 2.0


def test_emit_index_success_pairs_with_failure_for_rate_calculation():
    emitter = InMemoryMetricsEmitter()
    for _ in range(7):
        emit_index_success(emitter, modality=Modality.VECTOR)
    for _ in range(3):
        emit_index_failure(emitter, modality=Modality.VECTOR, error_kind="qdrant_5xx")

    success_key = (INDEX_SUCCESS_METRIC, (("modality", "vector"),))
    failure_key = (INDEX_FAILURE_METRIC, (("error_kind", "qdrant_5xx"), ("modality", "vector")))
    assert emitter.counters[success_key] == 7.0
    assert emitter.counters[failure_key] == 3.0


def test_emit_queue_depth_records_per_modality_gauge():
    emitter = InMemoryMetricsEmitter()
    emit_queue_depth(emitter, depth=42, modality=Modality.FULLTEXT)
    emit_queue_depth(emitter, depth=8, modality=Modality.SUMMARY)
    assert emitter.gauges[(QUEUE_DEPTH_METRIC, (("modality", "fulltext"),))] == 42.0
    assert emitter.gauges[(QUEUE_DEPTH_METRIC, (("modality", "summary"),))] == 8.0


def test_emit_worker_utilization_clamps_ratio_to_zero_one():
    emitter = InMemoryMetricsEmitter()
    emit_worker_utilization(emitter, busy=3, capacity=10, modality=Modality.VECTOR)
    key = (WORKER_UTILIZATION_METRIC, (("modality", "vector"),))
    assert emitter.gauges[key] == 0.3

    emit_worker_utilization(emitter, busy=15, capacity=10, modality=Modality.VECTOR)
    assert emitter.gauges[key] == 1.0, "utilization must clamp to 1.0 when busy > capacity"


def test_emit_worker_utilization_zero_capacity_emits_zero():
    """A zero-capacity pool (no workers configured for a modality)
    must not divide by zero — emit 0.0 instead."""
    emitter = InMemoryMetricsEmitter()
    emit_worker_utilization(emitter, busy=0, capacity=0, modality=Modality.GRAPH)
    key = (WORKER_UTILIZATION_METRIC, (("modality", "graph"),))
    assert emitter.gauges[key] == 0.0


def test_emit_worker_utilization_unscoped_when_modality_none():
    emitter = InMemoryMetricsEmitter()
    emit_worker_utilization(emitter, busy=5, capacity=10)
    key = (WORKER_UTILIZATION_METRIC, ())
    assert emitter.gauges[key] == 0.5


def test_noop_emitter_satisfies_protocol_without_recording():
    emitter = NoopMetricsEmitter()
    # Must not raise.
    emit_index_lag(emitter, seconds=99.0)
    emit_index_failure(emitter, modality=Modality.VECTOR)
    emit_queue_depth(emitter, depth=5)
    emit_worker_utilization(emitter, busy=1, capacity=2)


def test_metric_name_constants_are_indexing_prefixed():
    """The four §J.1 SLI names are exposed as constants so dashboards
    + alert rules can reference them by symbol. Lock the prefix
    convention to avoid accidental rename drift later."""
    for name in (
        INDEX_LAG_METRIC,
        INDEX_FAILURE_METRIC,
        INDEX_SUCCESS_METRIC,
        QUEUE_DEPTH_METRIC,
        WORKER_UTILIZATION_METRIC,
    ):
        assert name.startswith("indexing."), f"§J.1 SLI metric name must start with 'indexing.', got {name!r}"
