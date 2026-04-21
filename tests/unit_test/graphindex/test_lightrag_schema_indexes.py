from aperag.db.models import (
    LightRAGDocChunksModel,
    LightRAGVDBEntityModel,
    LightRAGVDBRelationModel,
)


def _index_map(model):
    return {index.name: index for index in model.__table__.indexes}


def test_lightrag_doc_chunks_has_workspace_full_doc_index():
    index = _index_map(LightRAGDocChunksModel)["idx_lightrag_doc_chunks_workspace_doc"]

    assert [column.name for column in index.columns] == ["workspace", "full_doc_id"]


def test_lightrag_vdb_entity_has_chunk_ids_gin_index():
    index = _index_map(LightRAGVDBEntityModel)["idx_lightrag_vdb_entity_chunk_ids_gin"]

    assert [column.name for column in index.columns] == ["chunk_ids"]
    assert index.dialect_options["postgresql"]["using"] == "gin"


def test_lightrag_vdb_relation_has_chunk_ids_gin_index():
    index = _index_map(LightRAGVDBRelationModel)["idx_lightrag_vdb_relation_chunk_ids_gin"]

    assert [column.name for column in index.columns] == ["chunk_ids"]
    assert index.dialect_options["postgresql"]["using"] == "gin"
