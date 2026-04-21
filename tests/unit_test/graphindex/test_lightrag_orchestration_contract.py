import asyncio

import pytest

from aperag.graph.lightrag.lightrag import LightRAG
from aperag.graph.lightrag.operate import extract_entities


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        return None

    def debug(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def log_extraction_progress(self, *_args, **_kwargs):
        return None

    def log_timing(self, *_args, **_kwargs):
        return None


def _build_rag():
    rag = LightRAG.__new__(LightRAG)
    rag.workspace = "workspace-1"
    rag.lightrag_logger = _FakeLogger()
    rag.chunk_entity_relation_graph = object()
    rag.entities_vdb = object()
    rag.relationships_vdb = object()
    rag.llm_model_func = object()
    rag.tokenizer = object()
    rag.llm_model_max_token_size = 4096
    rag.summary_to_max_tokens = 512
    rag.language = "English"
    rag.force_llm_summary_on_merge = 0
    rag.entity_extract_max_gleaning = 1
    rag.entity_types = ["organization", "person"]
    rag.example_number = None
    rag.llm_model_max_async = 4
    return rag


@pytest.mark.asyncio
async def test_grouping_process_chunk_results_returns_zero_counts_for_empty_components():
    rag = _build_rag()
    rag._find_connected_components = lambda _chunk_results: []

    result = await LightRAG._grouping_process_chunk_results(
        rag,
        chunk_results=[({}, {})],
        collection_id="collection-1",
    )

    assert result == {
        "groups_processed": 0,
        "total_entities": 0,
        "total_relations": 0,
        "collection_id": "collection-1",
    }


@pytest.mark.asyncio
async def test_grouping_process_chunk_results_keeps_current_serial_semantics(monkeypatch):
    rag = _build_rag()
    chunk_results = [
        (
            {"alpha": {"name": "alpha"}, "beta": {"name": "beta"}, "gamma": {"name": "gamma"}},
            {("alpha", "beta"): {"weight": 1}},
        )
    ]

    running = 0
    max_running = 0
    seen_components = []
    seen_filtered_nodes = []
    seen_filtered_edges = []

    async def _fake_merge_nodes_and_edges(*, chunk_results, component, **_kwargs):
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        seen_components.append(tuple(component))
        seen_filtered_nodes.append(sorted(chunk_results[0][0].keys()))
        seen_filtered_edges.append(sorted(chunk_results[0][1].keys()))
        await asyncio.sleep(0)
        running -= 1
        return {
            "entity_count": len(component),
            "relation_count": sum(len(edges) for _, edges in chunk_results),
        }

    monkeypatch.setattr("aperag.graph.lightrag.lightrag.merge_nodes_and_edges", _fake_merge_nodes_and_edges)

    result = await LightRAG._grouping_process_chunk_results(
        rag,
        chunk_results=chunk_results,
        collection_id="collection-1",
    )

    assert max_running == 1
    assert {frozenset(component) for component in seen_components} == {
        frozenset({"alpha", "beta"}),
        frozenset({"gamma"}),
    }
    assert {tuple(nodes) for nodes in seen_filtered_nodes} == {
        ("alpha", "beta"),
        ("gamma",),
    }
    assert {tuple(edges) for edges in seen_filtered_edges} == {
        (("alpha", "beta"),),
        (),
    }
    assert result == {
        "groups_processed": 2,
        "total_entities": 3,
        "total_relations": 1,
        "collection_id": "collection-1",
    }


@pytest.mark.asyncio
async def test_grouping_process_chunk_results_cancels_pending_components_on_first_exception(monkeypatch):
    rag = _build_rag()
    rag._find_connected_components = lambda _chunk_results: [["alpha"], ["beta"], ["gamma"]]
    chunk_results = [
        (
            {"alpha": {"name": "alpha"}, "beta": {"name": "beta"}, "gamma": {"name": "gamma"}},
            {},
        )
    ]

    entered_components = []
    beta_started = asyncio.Event()
    beta_cancelled = asyncio.Event()
    beta_completed = False
    gamma_completed = False

    async def _fake_merge_nodes_and_edges(*, component, **_kwargs):
        nonlocal beta_completed, gamma_completed
        entered_components.append(tuple(component))
        if component == ["alpha"]:
            raise RuntimeError("merge failed")
        if component == ["beta"]:
            beta_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                beta_cancelled.set()
                raise
        gamma_completed = True
        return {"entity_count": 1, "relation_count": 0}

    monkeypatch.setattr("aperag.graph.lightrag.lightrag.merge_nodes_and_edges", _fake_merge_nodes_and_edges)

    with pytest.raises(RuntimeError, match="merge failed"):
        await LightRAG._grouping_process_chunk_results(
            rag,
            chunk_results=chunk_results,
            collection_id="collection-1",
        )

    await asyncio.sleep(0)

    assert ("alpha",) in entered_components
    assert gamma_completed is False
    assert beta_completed is False
    assert beta_cancelled.is_set() or not beta_started.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunks", "expected_message"),
    [
        ({}, "No chunks provided for graph indexing"),
        ({"chunk-1": "not-a-dict"}, "Chunk chunk-1 is not a dictionary"),
        ({"chunk-1": {}}, "Chunk chunk-1 missing 'content' key"),
        ({"chunk-1": {"content": ""}}, "Chunk chunk-1 has empty content"),
    ],
)
async def test_aprocess_graph_indexing_validates_chunk_inputs(chunks, expected_message):
    rag = _build_rag()

    with pytest.raises(ValueError, match=expected_message):
        await LightRAG.aprocess_graph_indexing(rag, chunks, collection_id="collection-1")


@pytest.mark.asyncio
async def test_aprocess_graph_indexing_success_path_preserves_orchestration_contract(monkeypatch):
    rag = _build_rag()
    chunks = {
        "chunk-1": {"content": "alpha"},
        "chunk-2": {"content": "beta"},
    }
    captured = {}

    async def _fake_extract_entities(chunks, **kwargs):
        captured["extract_chunks"] = chunks
        captured["extract_kwargs"] = kwargs
        return [
            ({"alpha": {}, "beta": {}}, {("alpha", "beta"): {}}),
            ({"gamma": {}}, {}),
        ]

    async def _fake_grouping(chunk_results, collection_id=None):
        captured["grouping_chunk_results"] = chunk_results
        captured["grouping_collection_id"] = collection_id
        return {
            "groups_processed": 2,
            "total_entities": 99,
            "total_relations": 88,
            "collection_id": collection_id,
        }

    monkeypatch.setattr("aperag.graph.lightrag.lightrag.extract_entities", _fake_extract_entities)
    rag._grouping_process_chunk_results = _fake_grouping

    result = await LightRAG.aprocess_graph_indexing(rag, chunks, collection_id="collection-1")

    assert captured["extract_chunks"] == chunks
    assert captured["extract_kwargs"]["llm_model_max_async"] == rag.llm_model_max_async
    assert captured["grouping_collection_id"] == "collection-1"
    assert result == {
        "status": "success",
        "chunks_processed": 2,
        "entities_extracted": 3,
        "relations_extracted": 1,
        "groups_processed": 2,
        "collection_id": "collection-1",
    }


@pytest.mark.asyncio
async def test_aprocess_graph_indexing_re_raises_extract_entities_failure(monkeypatch):
    rag = _build_rag()
    chunks = {"chunk-1": {"content": "alpha"}}
    grouping_called = False

    async def _fake_extract_entities(*_args, **_kwargs):
        raise RuntimeError("extract failed")

    async def _fake_grouping(*_args, **_kwargs):
        nonlocal grouping_called
        grouping_called = True
        return {}

    monkeypatch.setattr("aperag.graph.lightrag.lightrag.extract_entities", _fake_extract_entities)
    rag._grouping_process_chunk_results = _fake_grouping

    with pytest.raises(RuntimeError, match="extract failed"):
        await LightRAG.aprocess_graph_indexing(rag, chunks, collection_id="collection-1")

    assert grouping_called is False


@pytest.mark.asyncio
async def test_aprocess_graph_indexing_re_raises_grouping_failure(monkeypatch):
    rag = _build_rag()
    chunks = {"chunk-1": {"content": "alpha"}}

    async def _fake_extract_entities(*_args, **_kwargs):
        return [({"alpha": {}}, {})]

    async def _fake_grouping(*_args, **_kwargs):
        raise RuntimeError("grouping failed")

    monkeypatch.setattr("aperag.graph.lightrag.lightrag.extract_entities", _fake_extract_entities)
    rag._grouping_process_chunk_results = _fake_grouping

    with pytest.raises(RuntimeError, match="grouping failed"):
        await LightRAG.aprocess_graph_indexing(rag, chunks, collection_id="collection-1")


@pytest.mark.asyncio
async def test_extract_entities_success_path_returns_stable_chunk_result_contract():
    async def _fake_llm(*_args, **_kwargs):
        return ""

    chunks = {
        "chunk-1": {
            "content": "plain chunk",
            "full_doc_id": "doc-1",
            "chunk_order_index": 0,
            "file_path": "/tmp/plain.txt",
        }
    }

    chunk_results = await extract_entities(
        chunks=chunks,
        use_llm_func=_fake_llm,
        entity_extract_max_gleaning=0,
        language="English",
        entity_types=["organization"],
        example_number=None,
        llm_model_max_async=1,
        lightrag_logger=_FakeLogger(),
    )

    assert isinstance(chunk_results, list)
    assert len(chunk_results) == 1

    nodes, edges = chunk_results[0]
    assert list(nodes.items()) == []
    assert list(edges.items()) == []


@pytest.mark.asyncio
async def test_extract_entities_cancels_pending_chunk_tasks_on_first_exception():
    started_slow_task = asyncio.Event()
    slow_task_cancelled = asyncio.Event()

    async def _fake_llm(prompt, **_kwargs):
        if "slow chunk" in prompt:
            started_slow_task.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                slow_task_cancelled.set()
                raise
        if "boom chunk" in prompt:
            await started_slow_task.wait()
            raise RuntimeError("llm failed")
        return ""

    chunks = {
        "chunk-boom": {
            "content": "boom chunk",
            "full_doc_id": "doc-1",
            "chunk_order_index": 0,
            "file_path": "/tmp/boom.txt",
        },
        "chunk-slow": {
            "content": "slow chunk",
            "full_doc_id": "doc-1",
            "chunk_order_index": 1,
            "file_path": "/tmp/slow.txt",
        },
    }

    with pytest.raises(RuntimeError, match="llm failed"):
        await extract_entities(
            chunks=chunks,
            use_llm_func=_fake_llm,
            entity_extract_max_gleaning=1,
            language="English",
            entity_types=["organization"],
            example_number=None,
            llm_model_max_async=2,
            lightrag_logger=_FakeLogger(),
        )

    assert slow_task_cancelled.is_set()
