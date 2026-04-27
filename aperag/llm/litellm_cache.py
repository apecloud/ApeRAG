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

"""Legacy LiteLLM cache compatibility.

ApeRAG owns caching in :mod:`aperag.cache`. This module remains as a
compatibility facade for older imports but intentionally does not enable
LiteLLM's built-in cache.
"""

import logging
from typing import Any, Dict

import litellm

from aperag.cache import application_cache_metrics

logger = logging.getLogger(__name__)


def setup_litellm_cache(*_args, **_kwargs) -> None:
    logger.info("LiteLLM built-in cache is disabled; ApeRAG application cache is authoritative")


def disable_litellm_cache() -> None:
    litellm.disable_cache()


def get_cache_stats() -> Dict[str, Any]:
    """
    Get local in-memory cache statistics for the current process.

    Returns:
        Dict containing cache statistics including hit rate calculation.
    """
    return {
        "cache_type": "aperag_application_cache",
        "namespaces": application_cache_metrics.snapshot(),
        "note": "LiteLLM built-in cache is disabled; stats come from aperag.cache metrics.",
    }


def clear_cache_stats() -> None:
    """Reset local in-memory cache statistics for the current process."""
    application_cache_metrics.clear()
    logger.info("Application cache statistics cleared")
