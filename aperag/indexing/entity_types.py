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

"""Collection graph entity-type string helpers.

Wave 11 keeps entity types as collection-local strings instead of a
registry table. This module owns the small amount of normalization,
i18n prompt mapping, and atomic config merge needed to keep that list
stable while allowing LLM-proposed types.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.domains.knowledge_base.db.models import Collection
from aperag.schema.common import KnowledgeGraphConfig
from aperag.schema.utils import dumpCollectionConfig, parseCollectionConfig

logger = logging.getLogger(__name__)


MAX_ENTITY_TYPE_CHARS = 64

_PROMPT_LANGUAGE_NAMES: dict[str, str] = {
    "zh-CN": "Chinese (Simplified)",
    "en-US": "English",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
}
_WHITESPACE_RE = re.compile(r"\s+")


def prompt_language_name(language: str | None) -> str:
    """Return the stable human-readable language name used in prompts."""
    return _PROMPT_LANGUAGE_NAMES.get(language or "", language or "Chinese (Simplified)")


def normalize_entity_type(raw: Any) -> str:
    """Clean one LLM/user entity-type string.

    This deliberately does not slugify, translate, hash, or semantically
    merge labels. The stored value is the collection-language display
    value. Overlong/empty values return ``""`` so callers can keep the
    entity row while avoiding config pollution.
    """
    value = _WHITESPACE_RE.sub(" ", str(raw or "").strip())
    if not value or len(value) > MAX_ENTITY_TYPE_CHARS:
        return ""
    return value


def merge_entity_type_values(current_types: Sequence[Any], new_types: Sequence[Any]) -> list[str]:
    """Merge two type lists with first-write-wins de-duplication."""
    merged: list[str] = []
    seen: set[str] = set()
    for raw in [*current_types, *new_types]:
        value = normalize_entity_type(raw)
        if not value:
            continue
        key = _dedupe_key(value)
        if key in seen:
            continue
        seen.add(key)
        merged.append(value)
    return merged


def collect_entity_types_from_entities(entities: Sequence[Any]) -> list[str]:
    """Return first-seen entity_type values from entity-like objects."""
    values: list[str] = []
    for entity in entities:
        values.append(getattr(entity, "entity_type", ""))
    return merge_entity_type_values((), values)


def format_entity_types_for_prompt(entity_types: Sequence[Any]) -> str:
    """Render every cleaned entity type for the prompt."""
    cleaned = merge_entity_type_values((), entity_types)
    return json.dumps(cleaned, ensure_ascii=False)


async def merge_entity_types(
    session: AsyncSession,
    collection_id: str,
    new_types: Sequence[str],
) -> list[str]:
    """Atomically merge new entity types into collection config.

    The production path passes a fresh ``AsyncSession`` and this helper
    opens the transaction. Tests may pass an already-open transaction;
    in that case we still issue the required ``SELECT ... FOR UPDATE``
    row lock and leave commit ownership to the caller.
    """
    normalized_new = merge_entity_type_values((), new_types)
    if not normalized_new:
        return []

    async with _session_transaction(session):
        stmt = select(Collection).where(Collection.id == collection_id).with_for_update()
        result = await session.execute(stmt)
        collection = result.scalar_one_or_none()
        if collection is None:
            logger.warning("entity type merge skipped: collection_id=%s not found", collection_id)
            return []

        config = parseCollectionConfig(collection.config)
        if config.knowledge_graph_config is None:
            config.knowledge_graph_config = KnowledgeGraphConfig(entity_types=[])

        current = config.knowledge_graph_config.entity_types or []
        merged = merge_entity_type_values(current, normalized_new)
        if merged != merge_entity_type_values((), current):
            config.knowledge_graph_config.entity_types = merged
            collection.config = dumpCollectionConfig(config)
            session.add(collection)
        return merged


@asynccontextmanager
async def _session_transaction(session: AsyncSession):
    if session.in_transaction():
        yield
    else:
        async with session.begin():
            yield


def _dedupe_key(value: str) -> str:
    if value.isascii():
        return value.lower()
    return value


__all__ = [
    "MAX_ENTITY_TYPE_CHARS",
    "collect_entity_types_from_entities",
    "format_entity_types_for_prompt",
    "merge_entity_type_values",
    "merge_entity_types",
    "normalize_entity_type",
    "prompt_language_name",
]
