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

"""Future-facing observability entrypoint for ApeRAG."""

from .config import ObservabilityConfig, ObservabilityMode, build_observability_config
from .context import (
    bind_observability_context,
    clear_observability_context,
    get_observability_context,
    reset_observability_context,
)
from .logging import configure_logging, make_logging_config
from .metrics import configure_metrics, get_meter, record_counter, record_histogram
from .tracing import (
    add_span_attributes,
    configure_fastapi,
    configure_process_observability,
    get_current_trace_info,
    get_tracer,
    start_span,
    trace_async_function,
    trace_function,
)

__all__ = [
    "ObservabilityConfig",
    "ObservabilityMode",
    "add_span_attributes",
    "bind_observability_context",
    "build_observability_config",
    "clear_observability_context",
    "configure_fastapi",
    "configure_logging",
    "configure_metrics",
    "configure_process_observability",
    "get_current_trace_info",
    "get_meter",
    "get_observability_context",
    "get_tracer",
    "make_logging_config",
    "record_counter",
    "record_histogram",
    "reset_observability_context",
    "start_span",
    "trace_async_function",
    "trace_function",
]
