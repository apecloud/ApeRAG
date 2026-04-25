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

"""Privacy helpers for logs and telemetry attributes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SENSITIVE_KEY_PARTS = (
    "authorization",
    "api_key",
    "apikey",
    "access_key",
    "secret",
    "password",
    "passwd",
    "pwd",
    "token",
    "cookie",
    "credential",
)

CONTENT_KEY_PARTS = (
    "prompt",
    "content",
    "document",
    "text",
    "message",
    "response",
    "completion",
)


def is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def is_content_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in CONTENT_KEY_PARTS)


def redact_value(key: str, value: Any, *, capture_content: bool = False) -> Any:
    """Redact secrets and bulky content from structured telemetry fields."""

    if value is None:
        return None
    if is_sensitive_key(key):
        return "[REDACTED]"
    if not capture_content and is_content_key(key):
        if isinstance(value, str):
            return {"redacted": True, "length": len(value)}
        if isinstance(value, (list, tuple, set, dict)):
            return {"redacted": True, "count": len(value)}
        return "[REDACTED]"
    return value


def sanitize_value(value: Any, *, capture_content: bool = False) -> Any:
    if isinstance(value, Mapping):
        return scrub_mapping(value, capture_content=capture_content)
    if isinstance(value, list):
        return [sanitize_value(item, capture_content=capture_content) for item in value]
    if isinstance(value, tuple):
        return [sanitize_value(item, capture_content=capture_content) for item in value]
    if isinstance(value, set):
        return [sanitize_value(item, capture_content=capture_content) for item in value]
    return value


def scrub_mapping(values: Mapping[str, Any], *, capture_content: bool = False) -> dict[str, Any]:
    return {
        key: sanitize_value(redact_value(key, value, capture_content=capture_content), capture_content=capture_content)
        for key, value in values.items()
    }


def sanitize_mapping(values: Mapping[str, Any], *, capture_content: bool = False) -> dict[str, Any]:
    return scrub_mapping(values, capture_content=capture_content)
