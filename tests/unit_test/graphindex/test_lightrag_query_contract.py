from types import SimpleNamespace

import pytest

from aperag.graph.lightrag import operate
from aperag.graph.lightrag.base import QueryParam
from aperag.graph.lightrag.lightrag import LightRAG


class _EmptyVectorStorage:
    cosine_better_than_threshold = 0.2

    async def query(self, *_args, **_kwargs):
        return []


@pytest.fixture
def rag_stub():
    rag = object.__new__(LightRAG)
    rag.chunk_entity_relation_graph = object()
    rag.entities_vdb = object()
    rag.relationships_vdb = object()
    rag.text_chunks = object()
    rag.tokenizer = SimpleNamespace(encode=lambda text: list(text or ""))
    rag.llm_model_func = None
    rag.language = "English"
    rag.example_number = None
    rag.chunks_vdb = object()
    return rag


@pytest.mark.asyncio
async def test_aquery_context_does_not_reuse_default_query_param(monkeypatch, rag_stub):
    seen_modes = []

    async def _fake_build_query_context(*args, **_kwargs):
        param = args[5]
        seen_modes.append(param.mode)
        param.mode = "local"
        return ([{"id": 1, "entity": "Alpha", "file_path": "/tmp/a.txt"}], [], [])

    monkeypatch.setattr("aperag.graph.lightrag.lightrag.build_query_context", _fake_build_query_context)

    await rag_stub.aquery_context("first query")
    await rag_stub.aquery_context("second query")

    assert seen_modes == ["global", "global"]


@pytest.mark.asyncio
async def test_aquery_context_returns_empty_string_for_empty_triplet(monkeypatch, rag_stub):
    async def _fake_build_query_context(*_args, **_kwargs):
        return [], [], []

    monkeypatch.setattr("aperag.graph.lightrag.lightrag.build_query_context", _fake_build_query_context)

    assert await rag_stub.aquery_context("no context") == ""


@pytest.mark.asyncio
async def test_build_query_context_returns_empty_triplet_for_empty_keywords(monkeypatch):
    async def _fake_get_keywords_from_query(*_args, **_kwargs):
        return [], []

    monkeypatch.setattr(operate, "get_keywords_from_query", _fake_get_keywords_from_query)

    result = await operate.build_query_context(
        "query",
        object(),
        object(),
        object(),
        object(),
        QueryParam(),
        SimpleNamespace(),
        None,
        language="English",
        example_number=None,
    )

    assert result == ([], [], [])


@pytest.mark.asyncio
async def test_build_query_context_falls_back_to_global_with_stable_triplet(monkeypatch):
    captured = {}

    async def _fake_get_keywords_from_query(*_args, **_kwargs):
        return ["high"], []

    async def _fake_build_from_keywords(*args, **_kwargs):
        query_param = args[6]
        captured["mode"] = query_param.mode
        return [], [], []

    monkeypatch.setattr(operate, "get_keywords_from_query", _fake_get_keywords_from_query)
    monkeypatch.setattr(operate, "_build_query_context_from_keywords", _fake_build_from_keywords)

    param = QueryParam(mode="hybrid")
    result = await operate.build_query_context(
        "query",
        object(),
        object(),
        object(),
        object(),
        param,
        SimpleNamespace(),
        None,
        language="English",
        example_number=None,
    )

    assert captured["mode"] == "global"
    assert result == ([], [], [])


@pytest.mark.asyncio
async def test_get_node_data_returns_empty_triplet_for_empty_results():
    result = await operate._get_node_data(
        "entity query",
        object(),
        _EmptyVectorStorage(),
        object(),
        QueryParam(),
        SimpleNamespace(),
    )

    assert result == ([], [], [])


@pytest.mark.asyncio
async def test_get_edge_data_returns_empty_triplet_for_empty_results():
    result = await operate._get_edge_data(
        "relation query",
        object(),
        _EmptyVectorStorage(),
        object(),
        QueryParam(),
        SimpleNamespace(),
    )

    assert result == ([], [], [])


@pytest.mark.asyncio
async def test_mix_mode_preserves_vector_only_text_hits(monkeypatch):
    async def _fake_get_node_data(*_args, **_kwargs):
        return [], [], []

    async def _fake_get_edge_data(*_args, **_kwargs):
        return [], [], []

    async def _fake_get_vector_context(*_args, **_kwargs):
        return [], [], [{"id": 1, "content": "vector-only chunk", "file_path": "/tmp/vector.txt"}]

    monkeypatch.setattr(operate, "_get_node_data", _fake_get_node_data)
    monkeypatch.setattr(operate, "_get_edge_data", _fake_get_edge_data)
    monkeypatch.setattr(operate, "_get_vector_context", _fake_get_vector_context)

    param = QueryParam(mode="mix")
    param.original_query = "vector query"

    entities_context, relations_context, text_units_context = await operate._build_query_context_from_keywords(
        "low",
        "high",
        object(),
        object(),
        object(),
        object(),
        param,
        SimpleNamespace(),
        chunks_vdb=object(),
    )

    assert entities_context == []
    assert relations_context == []
    assert [item["content"] for item in text_units_context] == ["vector-only chunk"]
