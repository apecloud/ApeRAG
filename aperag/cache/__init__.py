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

"""Cache primitives used by ApeRAG.

There are two intentionally separate cache families:

* read-primitive cache: parse-version-keyed MCP document read results;
* application cache: Redis-backed expensive provider/parser call results.
"""

from __future__ import annotations

from aperag.cache.application import (
    NAMESPACE_EMBEDDING,
    NAMESPACE_EMBEDDING_DIMENSION,
    NAMESPACE_LLM_COMPLETION,
    NAMESPACE_PARSER_PREFLIGHT,
    NAMESPACE_REMOTE_PARSER,
    NAMESPACE_RERANK,
    NAMESPACE_WEB_READ,
    NAMESPACE_WEB_SEARCH,
    ApplicationCache,
    ApplicationCacheMetrics,
    ApplicationCachePolicy,
    SyncApplicationCache,
    application_cache_metrics,
)
from aperag.cache.application_runtime import (
    application_cache_policy,
    get_application_cache,
    get_sync_application_cache,
    reset_application_cache_for_tests,
)
from aperag.cache.invalidation import invalidate_collection, invalidate_document
from aperag.cache.key import APPLICATION_CACHE_PREFIX, build_cache_key
from aperag.cache.parse_version_cache import (
    READ_PRIMITIVE_CACHE_PREFIX,
    L1Cache,
    L2Cache,
    NoopL2Cache,
)
from aperag.cache.read_primitive_cache import (
    ReadPrimitiveCache,
    build_read_primitive_cache,
)
from aperag.cache.runtime import (
    get_read_primitive_cache,
    reset_read_primitive_cache,
)

__all__ = [
    "APPLICATION_CACHE_PREFIX",
    "ApplicationCache",
    "ApplicationCacheMetrics",
    "ApplicationCachePolicy",
    "L1Cache",
    "L2Cache",
    "NAMESPACE_EMBEDDING",
    "NAMESPACE_EMBEDDING_DIMENSION",
    "NAMESPACE_LLM_COMPLETION",
    "NAMESPACE_PARSER_PREFLIGHT",
    "NAMESPACE_REMOTE_PARSER",
    "NAMESPACE_RERANK",
    "NAMESPACE_WEB_READ",
    "NAMESPACE_WEB_SEARCH",
    "NoopL2Cache",
    "READ_PRIMITIVE_CACHE_PREFIX",
    "ReadPrimitiveCache",
    "SyncApplicationCache",
    "application_cache_metrics",
    "application_cache_policy",
    "build_cache_key",
    "build_read_primitive_cache",
    "get_application_cache",
    "get_read_primitive_cache",
    "get_sync_application_cache",
    "invalidate_collection",
    "invalidate_document",
    "reset_application_cache_for_tests",
    "reset_read_primitive_cache",
]
