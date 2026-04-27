# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
from types import SimpleNamespace

from aperag.tasks.document import DocumentIndexTask


def test_upsert_graph_index_tolerates_graph_curation_invalidation_failure(monkeypatch, caplog):
    task = DocumentIndexTask()
    collection = SimpleNamespace(id="col-1")
    parsed = SimpleNamespace(content="doc content", file_path="/tmp/doc.txt")

    monkeypatch.setattr(
        "aperag.domains.knowledge_graph.graphindex.integration.run_index_document_sync",
        lambda **_kwargs: SimpleNamespace(
            doc_id="doc-1",
            chunks_created=2,
            entities_extracted=3,
            relations_extracted=4,
        ),
    )
    monkeypatch.setattr(
        "aperag.graph_curation.integration.run_expire_graph_curation_collection_sync",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("curation tables missing")),
    )

    with caplog.at_level(logging.WARNING):
        result = task._upsert_graph_index("doc-1", collection, parsed)

    assert result == {
        "status": "success",
        "doc_id": "doc-1",
        "chunks_created": 2,
        "entities_extracted": 3,
        "relations_extracted": 4,
    }
    assert "Graph curation invalidation failed for collection col-1 (document_reindex)" in caplog.text


def test_delete_graph_index_tolerates_graph_curation_invalidation_failure(monkeypatch, caplog):
    task = DocumentIndexTask()
    collection = SimpleNamespace(id="col-1")
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "aperag.domains.knowledge_graph.graphindex.integration.run_delete_document_sync",
        lambda **_kwargs: calls.append(("delete", _kwargs["doc_id"])),
    )
    monkeypatch.setattr(
        "aperag.graph_curation.integration.run_expire_graph_curation_collection_sync",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("curation db unavailable")),
    )

    with caplog.at_level(logging.WARNING):
        task._delete_graph_index("doc-2", collection)

    assert calls == [("delete", "doc-2")]
    assert "Graph curation invalidation failed for collection col-1 (document_delete)" in caplog.text
