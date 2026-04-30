from __future__ import annotations

import pytest

from aperag.domains.knowledge_base.service import document_service as document_service_module
from aperag.domains.knowledge_base.service.document_service import DocumentService


class _ScalarResult:
    def __init__(self, value: str | None):
        self._value = value

    def first(self):
        return self._value


class _ExecuteResult:
    def __init__(self, value: str | None):
        self._value = value

    def scalars(self):
        return _ScalarResult(self._value)


class _Session:
    async def execute(self, stmt):
        return _ExecuteResult("collections/col/documents/doc/derived/parse_v/chunks.jsonl")


class _DbOps:
    async def _execute_query(self, fn):
        return await fn(_Session())


class _Stream:
    def __init__(self, body: bytes):
        self._body = body

    async def __aiter__(self):
        yield self._body


class _ObjectStore:
    async def get(self, path: str):
        assert path == "collections/col/documents/doc/derived/parse_v/chunks.jsonl"
        body = (
            b'{"chunk_id":"c1","text":"first","section_path":"1","heading_anchor":"#first","page_idx":0}\n'
            b'{"chunk_id":"c2","text":"second","metadata":{"source":"doc.pdf"}}\n'
        )
        return _Stream(body), {}


@pytest.mark.asyncio
async def test_get_document_chunks_reads_serving_chunks_jsonl(monkeypatch):
    monkeypatch.setattr(document_service_module, "get_async_object_store", lambda: _ObjectStore())

    service = DocumentService()
    service.db_ops = _DbOps()

    chunks = await service.get_document_chunks("user", "col", "doc")

    assert [chunk.id for chunk in chunks] == ["c1", "c2"]
    assert [chunk.text for chunk in chunks] == ["first", "second"]
    assert chunks[0].metadata == {
        "section_path": "1",
        "heading_anchor": "#first",
        "page_idx": 0,
    }
    assert chunks[1].metadata == {"source": "doc.pdf"}
