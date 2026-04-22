from types import SimpleNamespace

from aperag.db.models import CollectionStatus, DocumentIndexType
from aperag.tasks.collection import CollectionTask
from aperag.tasks.document import DocumentIndexTask
from aperag.tasks.models import LocalDocumentInfo, ParsedDocumentData


def _parsed(document_id: str = "doc-1") -> ParsedDocumentData:
    return ParsedDocumentData(
        document_id=document_id,
        collection_id="col-1",
        content="hello world",
        doc_parts=[],
        file_path="/tmp/doc.txt",
        local_doc_info=LocalDocumentInfo(path="/tmp/doc.txt"),
    )


def _collection(collection_id: str = "col-1"):
    return SimpleNamespace(id=collection_id, config="{}", status=CollectionStatus.ACTIVE)


def test_create_graph_index_legacy_collection_keeps_legacy_truth_and_warms_v2_shadow(monkeypatch):
    task = DocumentIndexTask()
    collection = _collection()
    parsed = _parsed()
    calls = []

    monkeypatch.setattr("aperag.tasks.utils.get_document_and_collection", lambda document_id: (None, collection))
    monkeypatch.setattr("aperag.index.graph_index.graph_indexer", SimpleNamespace(is_enabled=lambda _c: True))
    monkeypatch.setattr("aperag.graphindex.integration.run_is_v2_initialized_sync", lambda collection_id: False)
    monkeypatch.setattr(
        "aperag.graph.lightrag_manager.process_document_for_celery",
        lambda **kwargs: calls.append(("legacy", kwargs))
        or {"status": "success", "doc_id": "doc-1", "chunks_created": 1},
    )
    monkeypatch.setattr(
        "aperag.graphindex.integration.run_index_document_sync",
        lambda **kwargs: calls.append(("v2", kwargs))
        or SimpleNamespace(doc_id="doc-1", chunks_created=2, entities_extracted=3, relations_extracted=4),
    )

    result = task.create_index("doc-1", DocumentIndexType.GRAPH.value, parsed)

    assert result.success is True
    assert result.data == {"status": "success", "doc_id": "doc-1", "chunks_created": 1}
    assert [name for name, _ in calls] == ["legacy", "v2"]


def test_update_graph_index_v2_collection_skips_legacy_mirror(monkeypatch):
    task = DocumentIndexTask()
    collection = _collection()
    parsed = _parsed()
    calls = []

    monkeypatch.setattr("aperag.tasks.utils.get_document_and_collection", lambda document_id: (None, collection))
    monkeypatch.setattr("aperag.index.graph_index.graph_indexer", SimpleNamespace(is_enabled=lambda _c: True))
    monkeypatch.setattr("aperag.graphindex.integration.run_is_v2_initialized_sync", lambda collection_id: True)
    monkeypatch.setattr(
        "aperag.graphindex.integration.run_index_document_sync",
        lambda **kwargs: calls.append(("v2", kwargs))
        or SimpleNamespace(doc_id="doc-1", chunks_created=2, entities_extracted=3, relations_extracted=4),
    )
    monkeypatch.setattr(
        "aperag.graph.lightrag_manager.process_document_for_celery",
        lambda **kwargs: calls.append(("legacy", kwargs)) or {"status": "success"},
    )

    result = task.update_index("doc-1", DocumentIndexType.GRAPH.value, parsed)

    assert result.success is True
    assert result.data == {
        "status": "success",
        "doc_id": "doc-1",
        "chunks_created": 2,
        "entities_extracted": 3,
        "relations_extracted": 4,
    }
    assert [name for name, _ in calls] == ["v2"]


def test_delete_graph_index_legacy_collection_deletes_truth_and_shadow(monkeypatch):
    task = DocumentIndexTask()
    collection = _collection()
    calls = []

    monkeypatch.setattr(
        "aperag.tasks.utils.get_document_and_collection", lambda document_id, ignore_deleted=False: (None, collection)
    )
    monkeypatch.setattr("aperag.index.graph_index.graph_indexer", SimpleNamespace(is_enabled=lambda _c: True))
    monkeypatch.setattr("aperag.graphindex.integration.run_is_v2_initialized_sync", lambda collection_id: False)
    monkeypatch.setattr(
        "aperag.graph.lightrag_manager.delete_document_for_celery",
        lambda **kwargs: calls.append(("legacy", kwargs)) or {"status": "success"},
    )
    monkeypatch.setattr(
        "aperag.graphindex.integration.run_delete_document_sync",
        lambda **kwargs: calls.append(("v2", kwargs)) or None,
    )

    result = task.delete_index("doc-1", DocumentIndexType.GRAPH.value)

    assert result.success is True
    assert [name for name, _ in calls] == ["legacy", "v2"]


def test_initialize_collection_marks_new_graph_collection_v2(monkeypatch):
    task = CollectionTask()
    collection = _collection("col-new")
    calls = []

    monkeypatch.setattr("aperag.tasks.collection.db_ops.query_collection_by_id", lambda collection_id: collection)
    monkeypatch.setattr("aperag.tasks.collection.db_ops.update_collection", lambda collection: None)
    monkeypatch.setattr(
        "aperag.tasks.collection.parseCollectionConfig", lambda cfg: SimpleNamespace(enable_knowledge_graph=True)
    )
    monkeypatch.setattr(task, "_initialize_vector_databases", lambda collection_id, collection: calls.append("vector"))
    monkeypatch.setattr(task, "_initialize_fulltext_index", lambda collection_id: calls.append("fulltext"))
    monkeypatch.setattr(
        "aperag.graphindex.integration.run_mark_collection_initialized_sync",
        lambda collection_id: calls.append(("mark_v2", collection_id)),
    )

    result = task.initialize_collection("col-new", 10)

    assert result.success is True
    assert ("mark_v2", "col-new") in calls
