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
from aperag.indexing.fulltext import (
    FulltextBackend,
    FulltextModality,
    InMemoryFulltextBackend,
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
    # Modalities (T1.3 / T1.4)
    "VectorModality",
    "VectorBackend",
    "InMemoryVectorBackend",
    "FulltextModality",
    "FulltextBackend",
    "InMemoryFulltextBackend",
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
]
