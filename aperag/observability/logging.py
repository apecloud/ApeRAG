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

import json
import logging
import sys
from datetime import datetime, timezone
from logging.config import dictConfig
from typing import Any, Dict

from .config import ObservabilityConfig
from .context import get_observability_context
from .privacy import sanitize_mapping, sanitize_value

_STANDARD_LOG_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class ObservabilityJsonFormatter(logging.Formatter):
    """Stable JSON log formatter used by API, worker, and beat processes."""

    def __init__(self, *, service_name: str, service_version: str, environment: str):
        super().__init__()
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        from .tracing import get_current_trace_info

        trace_id, span_id = get_current_trace_info()
        context = get_observability_context()

        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service.name": self.service_name,
            "service.version": self.service_version,
            "deployment.environment": self.environment,
            "trace_id": trace_id,
            "span_id": span_id,
            "request_id": context.request_id,
            "task_id": context.task_id,
            "domain": context.domain,
            "operation": context.operation,
            "outcome": getattr(record, "outcome", None),
        }

        if record.exc_info:
            exc_type = record.exc_info[0]
            exc_value = record.exc_info[1]
            payload["error.type"] = exc_type.__name__ if exc_type else None
            payload["error.message"] = sanitize_value(str(exc_value)) if exc_value else None
            payload["exception"] = self.formatException(record.exc_info)
        else:
            payload["error.type"] = getattr(record, "error_type", None)
            payload["error.message"] = sanitize_value(getattr(record, "error_message", None))

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_FIELDS
            and key not in payload
            and key not in {"outcome", "error_type", "error_message"}
            and not key.startswith("_")
        }
        if extras:
            payload["attributes"] = sanitize_mapping(extras)

        return json.dumps(
            {key: sanitize_value(value) for key, value in payload.items()}, ensure_ascii=False, default=str
        )


class ObservabilityTextFormatter(logging.Formatter):
    """Human-friendly formatter that still carries correlation identifiers."""

    def __init__(self, *, service_name: str):
        super().__init__(
            fmt=(
                "%(asctime)s - %(levelname)s - %(name)s - "
                f"service={service_name} trace_id=%(otel_trace_id)s span_id=%(otel_span_id)s - %(message)s"
            )
        )

    def format(self, record: logging.LogRecord) -> str:
        from .tracing import get_current_trace_info

        trace_id, span_id = get_current_trace_info()
        record.otel_trace_id = trace_id or ""
        record.otel_span_id = span_id or ""
        return super().format(record)


def make_logging_config(config: ObservabilityConfig) -> dict:
    formatter_name = "observability_json" if config.log_format == "json" else "observability_text"
    formatter_config: dict[str, Any]
    if config.log_format == "json":
        formatter_config = {
            "()": ObservabilityJsonFormatter,
            "service_name": config.service_name,
            "service_version": config.service_version,
            "environment": config.environment,
        }
    else:
        formatter_config = {"()": ObservabilityTextFormatter, "service_name": config.service_name}

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            formatter_name: formatter_config,
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": formatter_name,
                "level": "INFO",
                "stream": sys.stdout,
            }
        },
        "loggers": {
            "LiteLLM": {"level": "WARNING", "handlers": ["console"], "propagate": False},
            "aperag": {"level": "INFO", "propagate": True},
            "config": {"level": "INFO", "propagate": True},
            "celery": {"level": "INFO", "propagate": True},
            "celery.task": {"level": "INFO", "propagate": True},
            "celery.worker": {"level": "INFO", "propagate": True},
            "uvicorn": {"level": "INFO", "handlers": ["console"], "propagate": False},
            "uvicorn.error": {"level": "INFO", "handlers": ["console"], "propagate": False},
            "uvicorn.access": {"level": "INFO", "handlers": ["console"], "propagate": False},
        },
        "root": {"level": "INFO", "handlers": ["console"]},
    }


def configure_logging(config: ObservabilityConfig) -> None:
    dictConfig(make_logging_config(config))


build_logging_config = make_logging_config
