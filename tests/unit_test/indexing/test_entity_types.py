from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from aperag.indexing.entity_types import (
    format_entity_types_for_prompt,
    merge_entity_type_values,
    merge_entity_types,
    normalize_entity_type,
    prompt_language_name,
)
from aperag.schema.common import CollectionConfig, KnowledgeGraphConfig
from aperag.schema.utils import dumpCollectionConfig


def test_knowledge_graph_config_defaults_to_no_entity_types():
    assert KnowledgeGraphConfig().entity_types == []
    assert CollectionConfig().knowledge_graph_config.entity_types == []


def test_prompt_language_name_maps_collection_language():
    assert prompt_language_name("zh-CN") == "Chinese (Simplified)"
    assert prompt_language_name("en-US") == "English"


def test_normalize_entity_type_keeps_entity_safe_empty_for_bad_type():
    assert normalize_entity_type("  Medical   Condition  ") == "Medical Condition"
    assert normalize_entity_type("") == ""
    assert normalize_entity_type("x" * 65) == ""


def test_merge_entity_type_values_uses_first_write_wins_for_ascii_case():
    merged = merge_entity_type_values(["Person", "疾病"], [" person ", "疾病", "药物", "x" * 65])
    assert merged == ["Person", "疾病", "药物"]


def test_format_entity_types_for_prompt_allows_empty_list():
    assert format_entity_types_for_prompt([]) == "[]"
    assert format_entity_types_for_prompt(["人物", "组织"]) == '["人物", "组织"]'


def test_format_entity_types_for_prompt_does_not_cap_type_list():
    entity_types = [f"Type{i}" for i in range(75)]
    rendered = json.loads(format_entity_types_for_prompt(entity_types))
    assert rendered == entity_types


@pytest.mark.asyncio
async def test_merge_entity_types_uses_for_update_and_preserves_config():
    config = CollectionConfig(
        language="zh-CN",
        enable_summary=True,
        knowledge_graph_config=KnowledgeGraphConfig(entity_types=["Person", "疾病"]),
    )
    collection = _FakeCollection(config=dumpCollectionConfig(config))
    session = _FakeSession(collection)

    merged = await merge_entity_types(session, "col_w11", [" person ", "药物", "药物"])

    assert session.saw_for_update is True
    assert session.added is collection
    assert merged == ["Person", "疾病", "药物"]
    dumped = json.loads(collection.config)
    assert dumped["language"] == "zh-CN"
    assert dumped["enable_summary"] is True
    assert dumped["knowledge_graph_config"]["entity_types"] == ["Person", "疾病", "药物"]


@pytest.mark.asyncio
async def test_merge_entity_types_serializes_concurrent_updates():
    config = CollectionConfig(
        language="zh-CN",
        knowledge_graph_config=KnowledgeGraphConfig(entity_types=["人物"]),
    )
    row = _SharedCollectionRow(config=dumpCollectionConfig(config))

    session_a = _FakeSession(row)
    session_b = _FakeSession(row)

    task_a = asyncio.create_task(merge_entity_types(session_a, "col_w11", ["组织", "person"]))
    await asyncio.sleep(0)
    task_b = asyncio.create_task(merge_entity_types(session_b, "col_w11", ["疾病", "Person", "药物"]))
    merged_a, merged_b = await asyncio.gather(task_a, task_b)

    assert session_a.saw_for_update is True
    assert session_b.saw_for_update is True
    assert row.max_active_transactions == 1
    assert merged_a == ["人物", "组织", "person"]
    assert merged_b == ["人物", "组织", "person", "疾病", "药物"]
    dumped = json.loads(row.collection.config)
    assert dumped["knowledge_graph_config"]["entity_types"] == ["人物", "组织", "person", "疾病", "药物"]


class _FakeCollection:
    id = "col_w11"

    def __init__(self, *, config: str) -> None:
        self.config = config


class _FakeResult:
    def __init__(self, collection: _FakeCollection) -> None:
        self._collection = collection

    def scalar_one_or_none(self):
        return self._collection


class _SharedCollectionRow:
    def __init__(self, *, config: str) -> None:
        self.collection = _FakeCollection(config=config)
        self._lock = asyncio.Lock()
        self._active_transactions = 0
        self.max_active_transactions = 0

    async def acquire_for_update(self) -> None:
        await self._lock.acquire()
        self._active_transactions += 1
        self.max_active_transactions = max(self.max_active_transactions, self._active_transactions)
        await asyncio.sleep(0)

    def release_for_update(self) -> None:
        self._active_transactions -= 1
        self._lock.release()


class _FakeSession:
    def __init__(self, collection_or_row: _FakeCollection | _SharedCollectionRow) -> None:
        if isinstance(collection_or_row, _SharedCollectionRow):
            self._row = collection_or_row
            self._collection = collection_or_row.collection
        else:
            self._row = None
            self._collection = collection_or_row
        self.saw_for_update = False
        self.added = None
        self._locked_row = None

    def in_transaction(self) -> bool:
        return False

    @asynccontextmanager
    async def begin(self):
        try:
            yield
        finally:
            if self._locked_row is not None:
                self._locked_row.release_for_update()
                self._locked_row = None

    async def execute(self, stmt):
        self.saw_for_update = getattr(stmt, "_for_update_arg", None) is not None
        if self._row is not None:
            await self._row.acquire_for_update()
            self._locked_row = self._row
        return _FakeResult(self._collection)

    def add(self, item):
        self.added = item
