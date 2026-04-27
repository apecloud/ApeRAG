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

"""Phase celery indexing redesign — public surface.

Per ``docs/modularization/indexing-redesign-design-pack.md``. T1.1
Foundation lands the schema (``DocumentIndex`` + Modality / IndexStatus
enums), the ``ModalityWorker`` ABC that downstream T1.2-T1.5 modality
files implement, the atomic-write object-store helpers (§C.7), and a
deterministic parser entry point that produces the shared
``markdown.md`` / ``outline.json`` / ``chunks.jsonl`` artifacts
(§C.1 / §C.6).
"""

from aperag.indexing.base import DeriveResult, ModalityWorker
from aperag.indexing.cleanup import (
    CLEANUP_INTERVAL_SECONDS,
    ORPHAN_COOLDOWN_SECONDS,
    cleanup_for_deleted_collections,
    cleanup_for_deleted_documents,
    cleanup_orphan_parse_versions,
    find_orphan_parse_versions,
    run_cleanup_loop,
)
from aperag.indexing.dispatcher import (
    DEFAULT_MODALITIES,
    DispatchRequest,
    IndexingMode,
    all_modalities,
    dispatch_indexing,
    modalities_for_collection,
)
from aperag.indexing.fulltext import (
    FulltextBackend,
    FulltextModality,
    InMemoryFulltextBackend,
)
from aperag.indexing.graph import (
    KG_ARTIFACT_FILENAME,
    DescriptionPart,
    EntityLock,
    EntityRecord,
    EntityWithLineage,
    GraphExtractor,
    GraphModalityWorker,
    InMemoryEntityLock,
    InMemoryLineageGraphStore,
    LineageGraphStore,
    LineageMember,
    RedisEntityLock,
    RelationRecord,
    RelationWithLineage,
    parse_kg_jsonl,
    serialize_kg_jsonl,
)

# Wave 3 T3.1 chunk 2: ``aperag.indexing.keyword_extract`` is no longer
# re-exported here — eager import pulled the LLM completion stack into
# the indexing package's __init__, which transitively touched
# ``aperag.db.ops`` mid-load and triggered three unrelated circular
# imports during the hard-cut. The two production callers
# (``aperag/domains/retrieval/pipeline.py`` +
# ``aperag/service/search_pipeline_service.py``) already import directly
# from ``aperag.indexing.keyword_extract`` so the re-export was dead.
from aperag.indexing.limits import (
    EMBEDDING_CALL_TIMEOUT_SECONDS,
    LLM_CALL_TIMEOUT_SECONDS,
    UPLOAD_MAX_BYTES,
    bulkhead_timeout,
    reject_if_oversize,
)
from aperag.indexing.models import DocumentIndex, IndexStatus, Modality
from aperag.indexing.object_store import (
    InMemoryObjectStore,
    derived_artifact,
    derived_dir,
    read_or_none,
    read_or_none_async,
    source_artifact,
    write_atomic,
    write_atomic_async,
)
from aperag.indexing.observability import (
    INDEX_FAILURE_METRIC,
    INDEX_LAG_METRIC,
    INDEX_SUCCESS_METRIC,
    QUEUE_DEPTH_METRIC,
    WORKER_UTILIZATION_METRIC,
    InMemoryMetricsEmitter,
    MetricsEmitter,
    NoopMetricsEmitter,
    emit_index_failure,
    emit_index_lag,
    emit_index_success,
    emit_queue_depth,
    emit_worker_utilization,
)
from aperag.indexing.orchestrator import (
    HEARTBEAT_INTERVAL_SECONDS,
    INITIAL_RETRY_DELAY_SECONDS,
    MAX_RETRY_COUNT,
    DispatchPayload,
    InMemoryWorkQueue,
    ModalityWorkerFactory,
    OrchestratorConfig,
    RedisWorkQueue,
    WorkQueue,
    drain_queue_sync,
    process_one_task,
    run_fulltext_worker,
    run_graph_worker,
    run_summary_worker,
    run_vector_worker,
    run_vision_worker,
    run_worker_loop,
)
from aperag.indexing.parser import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PARSER_PIPELINE,
    ChunkingConfig,
    ParseConfig,
    ParseResult,
    parse_document,
    read_chunks,
)
from aperag.indexing.quota import (
    DEFAULT_TENANT_FALLBACK,
    InMemoryQuotaBackend,
    QuotaBackend,
    QuotaPolicy,
    QuotaPolicyRegistry,
    RedisQuotaBackend,
)
from aperag.indexing.reconciler import (
    HEARTBEAT_STALE_SECONDS,
    RECONCILE_BATCH_SIZE,
    RECONCILE_INTERVAL_SECONDS,
    reconcile_failed_retry,
    reconcile_pending_dispatch,
    reconcile_running_reclaim,
    run_reconcile_loop,
)
from aperag.indexing.summary import (
    InMemorySummaryBackend,
    SummaryBackend,
    SummaryModality,
)
from aperag.indexing.vector import (
    InMemoryVectorBackend,
    VectorBackend,
    VectorModality,
)
from aperag.indexing.vision import (
    InMemoryVisionBackend,
    VisionBackend,
    VisionModality,
)

__all__ = [
    # Schema
    "DocumentIndex",
    "Modality",
    "IndexStatus",
    # ABC
    "ModalityWorker",
    "DeriveResult",
    # Object store helpers
    "derived_dir",
    "derived_artifact",
    "source_artifact",
    "write_atomic",
    "write_atomic_async",
    "read_or_none",
    "read_or_none_async",
    "InMemoryObjectStore",
    # Parser
    "ChunkingConfig",
    "ParseConfig",
    "ParseResult",
    "parse_document",
    "read_chunks",
    "DEFAULT_PARSER_PIPELINE",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    # Modalities (T1.2 graph / T1.3 / T1.4)
    "VectorModality",
    "VectorBackend",
    "InMemoryVectorBackend",
    "FulltextModality",
    "FulltextBackend",
    "InMemoryFulltextBackend",
    "GraphModalityWorker",
    "LineageGraphStore",
    "InMemoryLineageGraphStore",
    "EntityLock",
    "InMemoryEntityLock",
    "RedisEntityLock",
    "LineageMember",
    "DescriptionPart",
    "EntityRecord",
    "RelationRecord",
    "EntityWithLineage",
    "RelationWithLineage",
    "GraphExtractor",
    "KG_ARTIFACT_FILENAME",
    "serialize_kg_jsonl",
    "parse_kg_jsonl",
    "SummaryModality",
    "SummaryBackend",
    "InMemorySummaryBackend",
    "VisionModality",
    "VisionBackend",
    "InMemoryVisionBackend",
    # Observability (T1.5)
    "MetricsEmitter",
    "NoopMetricsEmitter",
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
    # Orchestrator (T2.1)
    "DispatchPayload",
    "InMemoryWorkQueue",
    "RedisWorkQueue",
    "WorkQueue",
    "OrchestratorConfig",
    "ModalityWorkerFactory",
    "MAX_RETRY_COUNT",
    "INITIAL_RETRY_DELAY_SECONDS",
    "HEARTBEAT_INTERVAL_SECONDS",
    "process_one_task",
    "run_worker_loop",
    "run_vector_worker",
    "run_fulltext_worker",
    "run_graph_worker",
    "run_summary_worker",
    "run_vision_worker",
    "drain_queue_sync",
    # Reconciler (T2.1)
    "RECONCILE_INTERVAL_SECONDS",
    "RECONCILE_BATCH_SIZE",
    "HEARTBEAT_STALE_SECONDS",
    "reconcile_pending_dispatch",
    "reconcile_failed_retry",
    "reconcile_running_reclaim",
    "run_reconcile_loop",
    # Cleanup (T2.1 + T3.1 path C)
    "CLEANUP_INTERVAL_SECONDS",
    "ORPHAN_COOLDOWN_SECONDS",
    "find_orphan_parse_versions",
    "cleanup_orphan_parse_versions",
    "cleanup_for_deleted_documents",
    "cleanup_for_deleted_collections",
    "run_cleanup_loop",
    # Dispatcher (T3.1)
    "DispatchRequest",
    "IndexingMode",
    "DEFAULT_MODALITIES",
    "dispatch_indexing",
    "modalities_for_collection",
    "all_modalities",
    # Quota (T2.2 §H.5)
    "DEFAULT_TENANT_FALLBACK",
    "QuotaPolicy",
    "QuotaPolicyRegistry",
    "QuotaBackend",
    "InMemoryQuotaBackend",
    "RedisQuotaBackend",
    # Bulkhead limits (T2.2 §H.6)
    "LLM_CALL_TIMEOUT_SECONDS",
    "EMBEDDING_CALL_TIMEOUT_SECONDS",
    "UPLOAD_MAX_BYTES",
    "bulkhead_timeout",
    "reject_if_oversize",
]
